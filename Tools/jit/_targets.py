"""Target-specific code generation, parsing, and processing."""

import asyncio
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile
import typing

import _llvm
import _schema
import _stencils
import _writer

if sys.version_info < (3, 11):
    raise RuntimeError("Building the JIT compiler requires Python 3.11 or newer!")

TOOLS_JIT_BUILD = pathlib.Path(__file__).resolve() # 当前脚本所在路径
TOOLS_JIT = TOOLS_JIT_BUILD.parent
TOOLS = TOOLS_JIT.parent
CPYTHON = TOOLS.parent
PYTHON_EXECUTOR_CASES_C_H = CPYTHON / "Python" / "executor_cases.c.h"
TOOLS_JIT_TEMPLATE_C = TOOLS_JIT / "template.c"


# 分别代表不同平台（COFF、ELF、MachO）的段（Section）和重定位（Relocation）类型
_S = typing.TypeVar("_S", _schema.COFFSection, _schema.ELFSection, _schema.MachOSection)
_R = typing.TypeVar(
    "_R", _schema.COFFRelocation, _schema.ELFRelocation, _schema.MachORelocation
)


"""
代表一个JIT编译的目标
"""
@dataclasses.dataclass
class _Target(typing.Generic[_S, _R]):
    triple: str
    _: dataclasses.KW_ONLY
    alignment: int = 1
    args: typing.Sequence[str] = ()
    ghccc: bool = False
    prefix: str = ""
    stable: bool = False
    debug: bool = False
    verbose: bool = False

    """
    计算目标的哈希值
    """
    def _compute_digest(self, out: pathlib.Path) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.triple.encode())
        hasher.update(self.debug.to_bytes())
        # These dependencies are also reflected in _JITSources in regen.targets:
        hasher.update(PYTHON_EXECUTOR_CASES_C_H.read_bytes())
        hasher.update((out / "pyconfig.h").read_bytes())
        for dirpath, _, filenames in sorted(os.walk(TOOLS_JIT)):
            for filename in filenames:
                hasher.update(pathlib.Path(dirpath, filename).read_bytes())
        return hasher.hexdigest()

    """
    解析目标文件，并从中构建一个_stencils.StencilGroup对象
    """
    async def _parse(self, path: pathlib.Path) -> _stencils.StencilGroup:
        # 利用llvm-objdump获取目标文件的源代码（以汇编形式）
        group = _stencils.StencilGroup()
        args = ["--disassemble", "--reloc", f"{path}"]
        output = await _llvm.maybe_run("llvm-objdump", args, echo=self.verbose)
        # 将反汇编的结果存入StencilGroup.code.disassembly中
        if output is not None:
            group.code.disassembly.extend(
                line.expandtabs().strip()
                for line in output.splitlines()
                if not line.isspace()
            )
        # 利用llvm-readobj分析不同平台的目标文件
        args = [
            "--elf-output-style=JSON", # ELF的特定选项，选择输出样式为JSON
            "--expand-relocs", # 以扩展的多行格式显示每个重定位
            # "--pretty-print",
            "--section-data", # 显示每个section的数据区
            "--section-relocations", # 显示每个section的重定位信息
            "--section-symbols", # 显示每个section的符号信息
            "--sections", # 显示所有节section信息
            f"{path}",
        ]
        output = await _llvm.run("llvm-readobj", args, echo=self.verbose)
        # 针对一些特殊的符号清理替换
        # --elf-output-style=JSON is only *slightly* broken on Mach-O...
        output = output.replace("PrivateExtern\n", "\n")
        output = output.replace("Extern\n", "\n")
        # ...and also COFF:
        output = output[output.index("[", 1, None) :]
        output = output[: output.rindex("]", None, -1) + 1]
        # 将输出解（JSON）解析成Python对象（包含section信息的列表）
        sections: list[dict[typing.Literal["Section"], _S]] = json.loads(output)
        # 处理每个section
        for wrapped_section in sections:
            self._handle_section(wrapped_section["Section"], group)
        # The trampoline's entry point is just named "_ENTRY", since on some
        # platforms we later assume that any function starting with "_JIT_" uses
        # the GHC calling convention:
        entry_symbol = "_JIT_ENTRY" if "_JIT_ENTRY" in group.symbols else "_ENTRY"
        assert group.symbols[entry_symbol] == (_stencils.HoleValue.CODE, 0)
        # 处理数据：加入StencilGroup.data.disassembly中
        if group.data.body:
            line = f"0: {str(bytes(group.data.body)).removeprefix('b')}"
            group.data.disassembly.append(line)
        # 处理重定位信息
        group.process_relocations(alignment=self.alignment)
        return group

    def _handle_section(self, section: _S, group: _stencils.StencilGroup) -> None:
        raise NotImplementedError(type(self))

    def _handle_relocation(
        self, base: int, relocation: _R, raw: bytes
    ) -> _stencils.Hole:
        raise NotImplementedError(type(self))

    async def _compile(
        self, opname: str, c: pathlib.Path, tempdir: pathlib.Path
    ) -> _stencils.StencilGroup:
        # "Compile" the trampoline to an empty stencil group if it's not needed:
        if opname == "trampoline" and not self.ghccc:
            return _stencils.StencilGroup()
        o = tempdir / f"{opname}.o"
        args = [
            f"--target={self.triple}",
            "-DPy_BUILD_CORE_MODULE",
            "-D_DEBUG" if self.debug else "-DNDEBUG",
            f"-D_JIT_OPCODE={opname}",
            "-D_PyJIT_ACTIVE",
            "-D_Py_JIT",
            "-I.",
            f"-I{CPYTHON / 'Include'}",
            f"-I{CPYTHON / 'Include' / 'internal'}",
            f"-I{CPYTHON / 'Include' / 'internal' / 'mimalloc'}",
            f"-I{CPYTHON / 'Python'}",
            "-O3",
            "-c",
            # This debug info isn't necessary, and bloats out the JIT'ed code.
            # We *may* be able to re-enable this, process it, and JIT it for a
            # nicer debugging experience... but that needs a lot more research:
            "-fno-asynchronous-unwind-tables",
            # Don't call built-in functions that we can't find or patch:
            "-fno-builtin",
            # Emit relaxable 64-bit calls/jumps, so we don't have to worry about
            # about emitting in-range trampolines for out-of-range targets.
            # We can probably remove this and emit trampolines in the future:
            "-fno-plt",
            # Don't call stack-smashing canaries that we can't find or patch:
            "-fno-stack-protector",
            "-std=c11",
            *self.args,
        ]
        if self.ghccc:
            # This is a bit of an ugly workaround, but it makes the code much
            # smaller and faster, so it's worth it. We want to use the GHC
            # calling convention, but Clang doesn't support it. So, we *first*
            # compile the code to LLVM IR, perform some text replacements on the
            # IR to change the calling convention(!), and then compile *that*.
            # Once we have access to Clang 19, we can get rid of this and use
            # __attribute__((preserve_none)) directly in the C code instead:
            ll = tempdir / f"{opname}.ll"
            args_ll = args + [
                # -fomit-frame-pointer is necessary because the GHC calling
                # convention uses RBP to pass arguments:
                "-S",
                "-emit-llvm",
                "-fomit-frame-pointer",
                "-o",
                f"{ll}",
                f"{c}",
            ]
            await _llvm.run("clang", args_ll, echo=self.verbose)
            ir = ll.read_text()
            # This handles declarations, definitions, and calls to named symbols
            # starting with "_JIT_":
            ir = re.sub(
                r"(((noalias|nonnull|noundef) )*ptr @_JIT_\w+\()", r"ghccc \1", ir
            )
            # This handles calls to anonymous callees, since anything with
            # "musttail" needs to use the same calling convention:
            ir = ir.replace("musttail call", "musttail call ghccc")
            # Sometimes *both* replacements happen at the same site, so fix it:
            ir = ir.replace("ghccc ghccc", "ghccc")
            ll.write_text(ir)
            args_o = args + ["-Wno-unused-command-line-argument", "-o", f"{o}", f"{ll}"]
        else:
            args_o = args + ["-o", f"{o}", f"{c}"]
        # 利用clang编译成目标文件
        await _llvm.run("clang", args_o, echo=self.verbose)
        # 由_parse函数解析目标文件，返回StencilGroup对象
        return await self._parse(o)

    """
    编译生成模板，返回一个字典：对于每个操作名，对应一个StencilGroup
    """
    async def _build_stencils(self) -> dict[str, _stencils.StencilGroup]:
        # 读取生成的tier2的字节码语义（C语言形式），并找到对应的操作名称opname（也即bytecode，例如LOAD_FAST等）
        generated_cases = PYTHON_EXECUTOR_CASES_C_H.read_text()
        opnames = sorted(re.findall(r"\n {8}case (\w+): \{\n", generated_cases))
        tasks = []
        # 启用多线程，为每个字节码，都启用一个线程执行编译任务（调用_compile函数进行编译）
        with tempfile.TemporaryDirectory() as tempdir:
            work = pathlib.Path(tempdir).resolve()
            async with asyncio.TaskGroup() as group:
                coro = self._compile("trampoline", TOOLS_JIT / "trampoline.c", work)
                tasks.append(group.create_task(coro, name="trampoline"))
                for opname in opnames:
                    coro = self._compile(opname, TOOLS_JIT_TEMPLATE_C, work)
                    tasks.append(group.create_task(coro, name=opname))
        # 将所有编译结果组成一个dict返回
        return {task.get_name(): task.result() for task in tasks}

    """
    构建jit_stencils.h文件（包含了生成的机器码模板）
    """
    def build(
        self, out: pathlib.Path, *, comment: str = "", force: bool = False
    ) -> None:
        """Build jit_stencils.h in the given directory."""
        if not self.stable:
            warning = f"JIT support for {self.triple} is still experimental!"
            request = "Please report any issues you encounter.".center(len(warning))
            outline = "=" * len(warning)
            print("\n".join(["", outline, warning, request, outline, ""]))
        digest = f"// {self._compute_digest(out)}\n"
        jit_stencils = out / "jit_stencils.h"
        if (
            not force
            and jit_stencils.exists()
            and jit_stencils.read_text().startswith(digest)
        ):
            return
        # 调用_build_stencils函数获取模板
        stencil_groups = asyncio.run(self._build_stencils())
        # 将模板写入jit_stencils.h中
        jit_stencils_new = out / "jit_stencils.h.new"
        try:
            with jit_stencils_new.open("w") as file:
                file.write(digest)
                if comment:
                    file.write(f"// {comment}\n")
                file.write("\n")
                for line in _writer.dump(stencil_groups):
                    file.write(f"{line}\n")
            try:
                jit_stencils_new.replace(jit_stencils)
            except FileNotFoundError:
                # another process probably already moved the file
                if not jit_stencils.is_file():
                    raise
        finally:
            jit_stencils_new.unlink(missing_ok=True)


class _COFF(
    _Target[_schema.COFFSection, _schema.COFFRelocation]
):  # pylint: disable = too-few-public-methods
    def _handle_section(
        self, section: _schema.COFFSection, group: _stencils.StencilGroup
    ) -> None:
        flags = {flag["Name"] for flag in section["Characteristics"]["Flags"]}
        if "SectionData" in section:
            section_data_bytes = section["SectionData"]["Bytes"]
        else:
            # Zeroed BSS data, seen with printf debugging calls:
            section_data_bytes = [0] * section["RawDataSize"]
        if "IMAGE_SCN_MEM_EXECUTE" in flags:
            value = _stencils.HoleValue.CODE
            stencil = group.code
        elif "IMAGE_SCN_MEM_READ" in flags:
            value = _stencils.HoleValue.DATA
            stencil = group.data
        else:
            return
        base = len(stencil.body)
        group.symbols[section["Number"]] = value, base
        stencil.body.extend(section_data_bytes)
        for wrapped_symbol in section["Symbols"]:
            symbol = wrapped_symbol["Symbol"]
            offset = base + symbol["Value"]
            name = symbol["Name"]
            name = name.removeprefix(self.prefix)
            if name not in group.symbols:
                group.symbols[name] = value, offset
        for wrapped_relocation in section["Relocations"]:
            relocation = wrapped_relocation["Relocation"]
            hole = self._handle_relocation(base, relocation, stencil.body)
            stencil.holes.append(hole)

    def _unwrap_dllimport(self, name: str) -> tuple[_stencils.HoleValue, str | None]:
        if name.startswith("__imp_"):
            name = name.removeprefix("__imp_")
            name = name.removeprefix(self.prefix)
            return _stencils.HoleValue.GOT, name
        name = name.removeprefix(self.prefix)
        return _stencils.symbol_to_value(name)

    def _handle_relocation(
        self, base: int, relocation: _schema.COFFRelocation, raw: bytes
    ) -> _stencils.Hole:
        match relocation:
            case {
                "Offset": offset,
                "Symbol": s,
                "Type": {"Name": "IMAGE_REL_I386_DIR32" as kind},
            }:
                offset += base
                value, symbol = self._unwrap_dllimport(s)
                addend = int.from_bytes(raw[offset : offset + 4], "little")
            case {
                "Offset": offset,
                "Symbol": s,
                "Type": {
                    "Name": "IMAGE_REL_AMD64_REL32" | "IMAGE_REL_I386_REL32" as kind
                },
            }:
                offset += base
                value, symbol = self._unwrap_dllimport(s)
                addend = (
                    int.from_bytes(raw[offset : offset + 4], "little", signed=True) - 4
                )
            case {
                "Offset": offset,
                "Symbol": s,
                "Type": {
                    "Name": "IMAGE_REL_ARM64_BRANCH26"
                    | "IMAGE_REL_ARM64_PAGEBASE_REL21"
                    | "IMAGE_REL_ARM64_PAGEOFFSET_12A"
                    | "IMAGE_REL_ARM64_PAGEOFFSET_12L" as kind
                },
            }:
                offset += base
                value, symbol = self._unwrap_dllimport(s)
                addend = 0
            case _:
                raise NotImplementedError(relocation)
        return _stencils.Hole(offset, kind, value, symbol, addend)


class _ELF(
    _Target[_schema.ELFSection, _schema.ELFRelocation]
):  # pylint: disable = too-few-public-methods
    def _handle_section(
        self, section: _schema.ELFSection, group: _stencils.StencilGroup
    ) -> None:
        section_type = section["Type"]["Name"]
        flags = {flag["Name"] for flag in section["Flags"]["Flags"]}
        if section_type == "SHT_RELA":
            assert "SHF_INFO_LINK" in flags, flags
            assert not section["Symbols"]
            value, base = group.symbols[section["Info"]]
            if value is _stencils.HoleValue.CODE:
                stencil = group.code
            else:
                assert value is _stencils.HoleValue.DATA
                stencil = group.data
            for wrapped_relocation in section["Relocations"]:
                relocation = wrapped_relocation["Relocation"]
                hole = self._handle_relocation(base, relocation, stencil.body)
                stencil.holes.append(hole)
        elif section_type == "SHT_PROGBITS":
            if "SHF_ALLOC" not in flags:
                return
            if "SHF_EXECINSTR" in flags:
                value = _stencils.HoleValue.CODE
                stencil = group.code
            else:
                value = _stencils.HoleValue.DATA
                stencil = group.data
            group.symbols[section["Index"]] = value, len(stencil.body)
            for wrapped_symbol in section["Symbols"]:
                symbol = wrapped_symbol["Symbol"]
                offset = len(stencil.body) + symbol["Value"]
                name = symbol["Name"]["Name"]
                name = name.removeprefix(self.prefix)
                group.symbols[name] = value, offset
            stencil.body.extend(section["SectionData"]["Bytes"])
            assert not section["Relocations"]
        else:
            assert section_type in {
                "SHT_GROUP",
                "SHT_LLVM_ADDRSIG",
                "SHT_NOTE",
                "SHT_NULL",
                "SHT_STRTAB",
                "SHT_SYMTAB",
            }, section_type

    def _handle_relocation(
        self, base: int, relocation: _schema.ELFRelocation, raw: bytes
    ) -> _stencils.Hole:
        symbol: str | None
        match relocation:
            case {
                "Addend": addend,
                "Offset": offset,
                "Symbol": {"Name": s},
                "Type": {
                    "Name": "R_AARCH64_ADR_GOT_PAGE"
                    | "R_AARCH64_LD64_GOT_LO12_NC"
                    | "R_X86_64_GOTPCREL"
                    | "R_X86_64_GOTPCRELX"
                    | "R_X86_64_REX_GOTPCRELX" as kind
                },
            }:
                offset += base
                s = s.removeprefix(self.prefix)
                value, symbol = _stencils.HoleValue.GOT, s
            case {
                "Addend": addend,
                "Offset": offset,
                "Symbol": {"Name": s},
                "Type": {"Name": kind},
            }:
                offset += base
                s = s.removeprefix(self.prefix)
                value, symbol = _stencils.symbol_to_value(s)
            case _:
                raise NotImplementedError(relocation)
        return _stencils.Hole(offset, kind, value, symbol, addend)


class _MachO(
    _Target[_schema.MachOSection, _schema.MachORelocation]
):  # pylint: disable = too-few-public-methods
    def _handle_section(
        self, section: _schema.MachOSection, group: _stencils.StencilGroup
    ) -> None:
        assert section["Address"] >= len(group.code.body)
        assert "SectionData" in section
        flags = {flag["Name"] for flag in section["Attributes"]["Flags"]}
        name = section["Name"]["Value"]
        name = name.removeprefix(self.prefix)
        if "Debug" in flags:
            return
        if "SomeInstructions" in flags:
            value = _stencils.HoleValue.CODE
            stencil = group.code
            start_address = 0
            group.symbols[name] = value, section["Address"] - start_address
        else:
            value = _stencils.HoleValue.DATA
            stencil = group.data
            start_address = len(group.code.body)
            group.symbols[name] = value, len(group.code.body)
        base = section["Address"] - start_address
        group.symbols[section["Index"]] = value, base
        stencil.body.extend(
            [0] * (section["Address"] - len(group.code.body) - len(group.data.body))
        )
        stencil.body.extend(section["SectionData"]["Bytes"])
        assert "Symbols" in section
        for wrapped_symbol in section["Symbols"]:
            symbol = wrapped_symbol["Symbol"]
            offset = symbol["Value"] - start_address
            name = symbol["Name"]["Name"]
            name = name.removeprefix(self.prefix)
            group.symbols[name] = value, offset
        assert "Relocations" in section
        for wrapped_relocation in section["Relocations"]:
            relocation = wrapped_relocation["Relocation"]
            hole = self._handle_relocation(base, relocation, stencil.body)
            stencil.holes.append(hole)

    def _handle_relocation(
        self, base: int, relocation: _schema.MachORelocation, raw: bytes
    ) -> _stencils.Hole:
        symbol: str | None
        match relocation:
            case {
                "Offset": offset,
                "Symbol": {"Name": s},
                "Type": {
                    "Name": "ARM64_RELOC_GOT_LOAD_PAGE21"
                    | "ARM64_RELOC_GOT_LOAD_PAGEOFF12" as kind
                },
            }:
                offset += base
                s = s.removeprefix(self.prefix)
                value, symbol = _stencils.HoleValue.GOT, s
                addend = 0
            case {
                "Offset": offset,
                "Symbol": {"Name": s},
                "Type": {"Name": "X86_64_RELOC_GOT" | "X86_64_RELOC_GOT_LOAD" as kind},
            }:
                offset += base
                s = s.removeprefix(self.prefix)
                value, symbol = _stencils.HoleValue.GOT, s
                addend = (
                    int.from_bytes(raw[offset : offset + 4], "little", signed=True) - 4
                )
            case {
                "Offset": offset,
                "Section": {"Name": s},
                "Type": {"Name": "X86_64_RELOC_SIGNED" as kind},
            } | {
                "Offset": offset,
                "Symbol": {"Name": s},
                "Type": {"Name": "X86_64_RELOC_BRANCH" | "X86_64_RELOC_SIGNED" as kind},
            }:
                offset += base
                s = s.removeprefix(self.prefix)
                value, symbol = _stencils.symbol_to_value(s)
                addend = (
                    int.from_bytes(raw[offset : offset + 4], "little", signed=True) - 4
                )
            case {
                "Offset": offset,
                "Section": {"Name": s},
                "Type": {"Name": kind},
            } | {
                "Offset": offset,
                "Symbol": {"Name": s},
                "Type": {"Name": kind},
            }:
                offset += base
                s = s.removeprefix(self.prefix)
                value, symbol = _stencils.symbol_to_value(s)
                addend = 0
            case _:
                raise NotImplementedError(relocation)
        return _stencils.Hole(offset, kind, value, symbol, addend)


def get_target(host: str) -> _COFF | _ELF | _MachO:
    """Build a _Target for the given host "triple" and options."""
    # ghccc currently crashes Clang when combined with musttail on aarch64. :(
    target: _COFF | _ELF | _MachO
    if re.fullmatch(r"aarch64-apple-darwin.*", host):
        target = _MachO(host, alignment=8, prefix="_")
    elif re.fullmatch(r"aarch64-pc-windows-msvc", host):
        args = ["-fms-runtime-lib=dll"]
        target = _COFF(host, alignment=8, args=args)
    elif re.fullmatch(r"aarch64-.*-linux-gnu", host):
        args = ["-fpic"]
        target = _ELF(host, alignment=8, args=args)
    elif re.fullmatch(r"i686-pc-windows-msvc", host):
        args = ["-DPy_NO_ENABLE_SHARED"]
        target = _COFF(host, args=args, ghccc=True, prefix="_")
    elif re.fullmatch(r"x86_64-apple-darwin.*", host):
        target = _MachO(host, ghccc=True, prefix="_")
    elif re.fullmatch(r"x86_64-pc-windows-msvc", host):
        args = ["-fms-runtime-lib=dll"]
        target = _COFF(host, args=args, ghccc=True)
    elif re.fullmatch(r"x86_64-.*-linux-gnu", host):
        args = ["-fpic"]
        target = _ELF(host, args=args, ghccc=True)
    else:
        raise ValueError(host)
    return target
