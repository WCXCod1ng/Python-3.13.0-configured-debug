"""Core data structures for compiled code templates."""

import dataclasses
import enum
import sys
import typing

import _schema


@enum.unique
class HoleValue(enum.Enum):
    """
    Different "base" values that can be patched into holes (usually combined with the
    address of a symbol and/or an addend).
    """

    # The base address of the machine code for the current uop (exposed as _JIT_ENTRY):
    CODE = enum.auto() # 当前指令的起始地址
    # The base address of the machine code for the next uop (exposed as _JIT_CONTINUE):
    CONTINUE = enum.auto() # 下一条指令的起始地址
    # The base address of the read-only data for this uop:
    DATA = enum.auto()
    # The address of the current executor (exposed as _JIT_EXECUTOR):
    EXECUTOR = enum.auto() # 当前执行器的地址，执行器负责在执行JIT编译后生成的机器代码
    # The base address of the "global" offset table located in the read-only data.
    # Shouldn't be present in the final stencils, since these are all replaced with
    # equivalent DATA values:
    GOT = enum.auto() # 全局偏移表的基地址，存储在只读数据中
    # The current uop's oparg (exposed as _JIT_OPARG):
    OPARG = enum.auto() # 当前uop的操作数
    # The current uop's operand on 64-bit platforms (exposed as _JIT_OPERAND):
    OPERAND = enum.auto()
    # The current uop's operand on 32-bit platforms (exposed as _JIT_OPERAND_HI/LO):
    OPERAND_HI = enum.auto()
    OPERAND_LO = enum.auto()
    # The current uop's target (exposed as _JIT_TARGET):
    TARGET = enum.auto() # 当前uop的目标地址，表示操作的目标
    # The base address of the machine code for the jump target (exposed as _JIT_JUMP_TARGET):
    JUMP_TARGET = enum.auto()
    # The base address of the machine code for the error jump target (exposed as _JIT_ERROR_TARGET):
    ERROR_TARGET = enum.auto()
    # The index of the exit to be jumped through (exposed as _JIT_EXIT_INDEX):
    EXIT_INDEX = enum.auto()
    # The base address of the machine code for the first uop (exposed as _JIT_TOP):
    TOP = enum.auto() # 第一个uop的机器代码的基地址
    # A hardcoded value of zero (used for symbol lookups):
    ZERO = enum.auto()


# TODO 将实际的重定位指令映射到jit.c中定义的那些patch函数上
# Map relocation types to our JIT's patch functions. "r" suffixes indicate that
# the patch function is relative. "x" suffixes indicate that they are "relaxing"
# (see comments in jit.c for more info):
_PATCH_FUNCS = {
    # aarch64-apple-darwin:
    "ARM64_RELOC_BRANCH26": "patch_aarch64_26r",
    "ARM64_RELOC_GOT_LOAD_PAGE21": "patch_aarch64_21rx",
    "ARM64_RELOC_GOT_LOAD_PAGEOFF12": "patch_aarch64_12x",
    "ARM64_RELOC_PAGE21": "patch_aarch64_21r",
    "ARM64_RELOC_PAGEOFF12": "patch_aarch64_12",
    "ARM64_RELOC_UNSIGNED": "patch_64",
    # x86_64-pc-windows-msvc:
    "IMAGE_REL_AMD64_REL32": "patch_x86_64_32rx",
    # aarch64-pc-windows-msvc:
    "IMAGE_REL_ARM64_BRANCH26": "patch_aarch64_26r",
    "IMAGE_REL_ARM64_PAGEBASE_REL21": "patch_aarch64_21rx",
    "IMAGE_REL_ARM64_PAGEOFFSET_12A": "patch_aarch64_12",
    "IMAGE_REL_ARM64_PAGEOFFSET_12L": "patch_aarch64_12x",
    # i686-pc-windows-msvc:
    "IMAGE_REL_I386_DIR32": "patch_32",
    "IMAGE_REL_I386_REL32": "patch_x86_64_32rx",
    # aarch64-unknown-linux-gnu:
    "R_AARCH64_ABS64": "patch_64",
    "R_AARCH64_ADD_ABS_LO12_NC": "patch_aarch64_12",
    "R_AARCH64_ADR_GOT_PAGE": "patch_aarch64_21rx",
    "R_AARCH64_ADR_PREL_PG_HI21": "patch_aarch64_21r",
    "R_AARCH64_CALL26": "patch_aarch64_26r",
    "R_AARCH64_JUMP26": "patch_aarch64_26r",
    "R_AARCH64_LD64_GOT_LO12_NC": "patch_aarch64_12x",
    "R_AARCH64_MOVW_UABS_G0_NC": "patch_aarch64_16a",
    "R_AARCH64_MOVW_UABS_G1_NC": "patch_aarch64_16b",
    "R_AARCH64_MOVW_UABS_G2_NC": "patch_aarch64_16c",
    "R_AARCH64_MOVW_UABS_G3": "patch_aarch64_16d",
    # x86_64-unknown-linux-gnu:
    "R_X86_64_64": "patch_64",
    "R_X86_64_GOTPCREL": "patch_32r",
    "R_X86_64_GOTPCRELX": "patch_x86_64_32rx",
    "R_X86_64_PC32": "patch_32r",
    "R_X86_64_REX_GOTPCRELX": "patch_x86_64_32rx",
    # x86_64-apple-darwin:
    "X86_64_RELOC_BRANCH": "patch_32r",
    "X86_64_RELOC_GOT": "patch_x86_64_32rx",
    "X86_64_RELOC_GOT_LOAD": "patch_x86_64_32rx",
    "X86_64_RELOC_SIGNED": "patch_32r",
    "X86_64_RELOC_UNSIGNED": "patch_64",
}
# TODO 将HoleValue的不同取值映射为C的不同表达式
# Translate HoleValues to C expressions:
_HOLE_EXPRS = {
    HoleValue.CODE: "(uintptr_t)code",
    HoleValue.CONTINUE: "(uintptr_t)code + sizeof(code_body)",
    HoleValue.DATA: "(uintptr_t)data",
    HoleValue.EXECUTOR: "(uintptr_t)executor",
    # These should all have been turned into DATA values by process_relocations:
    # HoleValue.GOT: "",
    HoleValue.OPARG: "instruction->oparg",
    HoleValue.OPERAND: "instruction->operand",
    HoleValue.OPERAND_HI: "(instruction->operand >> 32)",
    HoleValue.OPERAND_LO: "(instruction->operand & UINT32_MAX)",
    HoleValue.TARGET: "instruction->target",
    HoleValue.JUMP_TARGET: "instruction_starts[instruction->jump_target]",
    HoleValue.ERROR_TARGET: "instruction_starts[instruction->error_target]",
    HoleValue.EXIT_INDEX: "instruction->exit_index",
    HoleValue.TOP: "instruction_starts[1]",
    HoleValue.ZERO: "",
}


@dataclasses.dataclass
class Hole:
    """
    A "hole" in the stencil to be patched with a computed runtime value.

    Analogous to relocation records in an object file.
    """

    offset: int # 偏移，指明了该hole在机器码中的位置（相较于Stencil.body的偏移量），用于patch_*中加在code上的偏移，例如code + 0x4中的0x4
    kind: _schema.HoleKind
    # Patch with this base value:
    value: HoleValue
    # ...plus the address of this symbol:
    symbol: str | None # 如果该hole与某个符号（如函数、变量）相关，则该字段包含了符号的名称（没有则为null），例如_JIT_CONTINUE
    # ...plus this addend:
    addend: int # 加数，用来进一步调整patch的值，可以用于符号地址的偏移，或者对指令中的立即数操作
    func: str = dataclasses.field(init=False) # patch该hole所需要的patch函数（所有patch函数见jit.c）
    # Convenience method:
    replace = dataclasses.replace # 用于替换当前Hole实例中的字段，生成一个新的Hole实例

    """
    根据hole的类型选择合适的patch函数
    """
    def __post_init__(self) -> None:
        self.func = _PATCH_FUNCS[self.kind]

    def fold(self, other: typing.Self) -> typing.Self | None:
        """Combine two holes into a single hole, if possible."""
        if (
            self.offset + 4 == other.offset
            and self.value == other.value
            and self.symbol == other.symbol
            and self.addend == other.addend
            and self.func == "patch_aarch64_21rx"
            and other.func == "patch_aarch64_12x"
        ): # 尝试通过一个patch_aarch64_33rx替换patch_aarch64_21rx和patch_aarch64_12x
            # These can *only* be properly relaxed when they appear together and
            # patch the same value:
            folded = self.replace()
            folded.func = "patch_aarch64_33rx"
            return folded
        return None

    """
    为当前的Hole生成如同jit_stencils.h中类似于patch_32r(location, value)的语句（生成patch语句）
    下面的注释均以emit__FATL_ERROR为例
    """
    def as_c(self, where: str) -> str:
        """Dump this hole as a call to a patch_* function."""
        location = f"{where} + {self.offset:#x}" # location，例如code + 0x4
        value = _HOLE_EXPRS[self.value] # value，例如(uintptr_t)data，也就是HoleValue.DATA
        if self.symbol:
            if value:
                value += " + "
            value += f"(uintptr_t)&{self.symbol}" # 有符号则add上对应的地址，例如patch_64(data + 0x28, (uintptr_t)&_Py_FatalErrorFunc);
        if _signed(self.addend):
            if value:
                value += " + "
            value += f"{_signed(self.addend):#x}" # 有偏移量就加偏移量，例如 + -0x4
        return f"{self.func}({location}, {value});"


@dataclasses.dataclass
class Stencil:
    """
    A contiguous block of machine code or data to be copied-and-patched.

    Analogous to a section or segment in an object file.
    """

    # body存储了实际的机器代码或数据
    body: bytearray = dataclasses.field(default_factory=bytearray, init=False)
    # 这是一个列表，包含了许多Hole对象，也即需要patch的占位符
    holes: list[Hole] = dataclasses.field(default_factory=list, init=False)
    disassembly: list[str] = dataclasses.field(default_factory=list, init=False)

    def pad(self, alignment: int) -> None:
        """Pad the stencil to the given alignment."""
        offset = len(self.body)
        padding = -offset % alignment
        self.disassembly.append(f"{offset:x}: {' '.join(['00'] * padding)}")
        self.body.extend([0] * padding)

    """
    在AArch64架构下生成跳转指令的补丁
    """
    def emit_aarch64_trampoline(self, hole: Hole) -> None:
        """Even with the large code model, AArch64 Linux insists on 28-bit jumps."""
        base = len(self.body) # 当前Stencil中body的长度，也表示了机器码的结束位置
        where = slice(hole.offset, hole.offset + 4) # 指示hole在body中所占用的字节范围（4个字节）
        instruction = int.from_bytes(self.body[where], sys.byteorder) # 拿到对应位置的字节数据并解析成整数
        instruction &= 0xFC000000
        instruction |= ((base - hole.offset) >> 2) & 0x03FFFFFF # 将跳转偏移写到指令中
        self.body[where] = instruction.to_bytes(4, sys.byteorder) # 写回到body中
        self.disassembly += [
            f"{base + 4 * 0:x}: d2800008      mov     x8, #0x0",
            f"{base + 4 * 0:016x}:  R_AARCH64_MOVW_UABS_G0_NC    {hole.symbol}",
            f"{base + 4 * 1:x}: f2a00008      movk    x8, #0x0, lsl #16",
            f"{base + 4 * 1:016x}:  R_AARCH64_MOVW_UABS_G1_NC    {hole.symbol}",
            f"{base + 4 * 2:x}: f2c00008      movk    x8, #0x0, lsl #32",
            f"{base + 4 * 2:016x}:  R_AARCH64_MOVW_UABS_G2_NC    {hole.symbol}",
            f"{base + 4 * 3:x}: f2e00008      movk    x8, #0x0, lsl #48",
            f"{base + 4 * 3:016x}:  R_AARCH64_MOVW_UABS_G3       {hole.symbol}",
            f"{base + 4 * 4:x}: d61f0100      br      x8",
        ]
        for code in [
            0xD2800008.to_bytes(4, sys.byteorder),
            0xF2A00008.to_bytes(4, sys.byteorder),
            0xF2C00008.to_bytes(4, sys.byteorder),
            0xF2E00008.to_bytes(4, sys.byteorder),
            0xD61F0100.to_bytes(4, sys.byteorder),
        ]:
            self.body.extend(code) # 将跳转指令的机器码追加到body中
        for i, kind in enumerate(
            [
                "R_AARCH64_MOVW_UABS_G0_NC",
                "R_AARCH64_MOVW_UABS_G1_NC",
                "R_AARCH64_MOVW_UABS_G2_NC",
                "R_AARCH64_MOVW_UABS_G3",
            ]
        ):
            self.holes.append(hole.replace(offset=base + 4 * i, kind=kind))

    """
    从Stencil中移除一个特定的跳转指令（通常是一个零长度的继续跳转指令）
    """
    def remove_jump(self, *, alignment: int = 1) -> None:
        """Remove a zero-length continuation jump, if it exists."""
        hole = max(self.holes, key=lambda hole: hole.offset) # 找到Stencil中偏移量最大的那个hole（即最后一个hole）
        match hole: # 根据Hole的不同类型（不同跳转类型）来确定跳转指令
            case Hole(
                offset=offset,
                kind="IMAGE_REL_AMD64_REL32",
                value=HoleValue.GOT,
                symbol="_JIT_CONTINUE",
                addend=-4,
            ) as hole:
                # jmp qword ptr [rip]
                jump = b"\x48\xFF\x25\x00\x00\x00\x00"
                offset -= 3
            case Hole(
                offset=offset,
                kind="IMAGE_REL_I386_REL32" | "X86_64_RELOC_BRANCH",
                value=HoleValue.CONTINUE,
                symbol=None,
                addend=-4,
            ) as hole:
                # jmp 5
                jump = b"\xE9\x00\x00\x00\x00"
                offset -= 1
            case Hole(
                offset=offset,
                kind="R_AARCH64_JUMP26",
                value=HoleValue.CONTINUE,
                symbol=None,
                addend=0,
            ) as hole:
                # b #4
                jump = b"\x00\x00\x00\x14"
            case Hole(
                offset=offset,
                kind="R_X86_64_GOTPCRELX",
                value=HoleValue.GOT,
                symbol="_JIT_CONTINUE",
                addend=addend,
            ) as hole:
                assert _signed(addend) == -4
                # jmp qword ptr [rip]
                jump = b"\xFF\x25\x00\x00\x00\x00"
                offset -= 2
            case _:
                return
        if self.body[offset:] == jump and offset % alignment == 0: # 检查 Stencil.body 中从 offset 位置开始的字节序列是否等于我们刚刚匹配到的跳转指令（jump）。如果匹配，表示这个跳转指令确实存在于 Stencil.body 中；检查跳转指令是否对齐。如果指定了对齐要求，且跳转指令的偏移量符合对齐要求，则认为这个跳转指令是需要移除的
            self.body = self.body[:offset] # 如果跳转指令符合要求，则删除该指令及其之后的所有内容
            self.holes.remove(hole) # 移除对应的hole


@dataclasses.dataclass
class StencilGroup:
    """
    Code and data corresponding to a given micro-opcode.

    Analogous to an entire object file.
    """

    # 分别存储micro-opcode的机器码部分和数据部分（都是Stencil）
    code: Stencil = dataclasses.field(default_factory=Stencil, init=False)
    data: Stencil = dataclasses.field(default_factory=Stencil, init=False)
    # 符号名映射到一个元组，元组中包含HoleValue（代表patch时的基准值，如data），int则表示符号的附加偏移addend
    symbols: dict[int | str, tuple[HoleValue, int]] = dataclasses.field(
        default_factory=dict, init=False
    )
    # 字典，用于跟踪GOT的偏移量
    _got: dict[str, int] = dataclasses.field(default_factory=dict, init=False)

    def process_relocations(self, *, alignment: int = 1) -> None:
        """Fix up all GOT and internal relocations for this stencil group."""
        # 处理跳转指令
        for hole in self.code.holes.copy():
            if (
                hole.kind
                in {"R_AARCH64_CALL26", "R_AARCH64_JUMP26", "ARM64_RELOC_BRANCH26"}
                and hole.value is HoleValue.ZERO
            ):
                self.code.pad(alignment)
                self.code.emit_aarch64_trampoline(hole)
                self.code.holes.remove(hole)
        # 移除多余的jump指令以及对齐
        self.code.remove_jump(alignment=alignment)
        self.code.pad(alignment)
        self.data.pad(8)
        # 处理符号和更新空洞
        for stencil in [self.code, self.data]:
            for hole in stencil.holes:
                if hole.value is HoleValue.GOT: # 该hole是GOT的入口
                    assert hole.symbol is not None
                    hole.value = HoleValue.DATA
                    hole.addend += self._global_offset_table_lookup(hole.symbol)
                    hole.symbol = None
                elif hole.symbol in self.symbols: # 该hole是一个symbols的一项，直接使用指定内容
                    hole.value, addend = self.symbols[hole.symbol]
                    hole.addend += addend
                    hole.symbol = None
                elif (
                    hole.kind in {"IMAGE_REL_AMD64_REL32"}
                    and hole.value is HoleValue.ZERO
                ): # 异常
                    raise ValueError(
                        f"Add PyAPI_FUNC(...) or PyAPI_DATA(...) to declaration of {hole.symbol}!"
                    )
        # 生成GOT
        self._emit_global_offset_table()
        self.code.holes.sort(key=lambda hole: hole.offset)
        self.data.holes.sort(key=lambda hole: hole.offset)

    """
    返回符号在全局偏移表GOT中的偏移量
    """
    def _global_offset_table_lookup(self, symbol: str) -> int:
        return len(self.data.body) + self._got.setdefault(symbol, 8 * len(self._got))

    """
    用于生成GOT
    """
    def _emit_global_offset_table(self) -> None:
        got = len(self.data.body)
        for s, offset in self._got.items():
            if s in self.symbols:
                value, addend = self.symbols[s]
                symbol = None
            else:
                value, symbol = symbol_to_value(s)
                addend = 0
            self.data.holes.append(
                Hole(got + offset, "R_X86_64_64", value, symbol, addend)
            )
            value_part = value.name if value is not HoleValue.ZERO else ""
            if value_part and not symbol and not addend:
                addend_part = ""
            else:
                signed = "+" if symbol is not None else ""
                addend_part = f"&{symbol}" if symbol else ""
                addend_part += f"{_signed(addend):{signed}#x}"
                if value_part:
                    value_part += "+"
            self.data.disassembly.append(
                f"{len(self.data.body):x}: {value_part}{addend_part}"
            )
            self.data.body.extend([0] * 8)

    def as_c(self, opname: str) -> str:
        """Dump this hole as a StencilGroup initializer."""
        return f"{{emit_{opname}, {len(self.code.body)}, {len(self.data.body)}}}"


def symbol_to_value(symbol: str) -> tuple[HoleValue, str | None]:
    """
    Convert a symbol name to a HoleValue and a symbol name.

    Some symbols (starting with "_JIT_") are special and are converted to their
    own HoleValues.
    """
    if symbol.startswith("_JIT_"):
        try:
            return HoleValue[symbol.removeprefix("_JIT_")], None
        except KeyError:
            pass
    return HoleValue.ZERO, symbol


def _signed(value: int) -> int:
    value %= 1 << 64
    if value & (1 << 63):
        value -= 1 << 64
    return value
