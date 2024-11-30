#ifdef _Py_JIT

#include "Python.h"

#include "pycore_abstract.h"
#include "pycore_call.h"
#include "pycore_ceval.h"
#include "pycore_critical_section.h"
#include "pycore_dict.h"
#include "pycore_intrinsics.h"
#include "pycore_long.h"
#include "pycore_opcode_metadata.h"
#include "pycore_opcode_utils.h"
#include "pycore_optimizer.h"
#include "pycore_pyerrors.h"
#include "pycore_setobject.h"
#include "pycore_sliceobject.h"
#include "pycore_jit.h"

// Memory management stuff: ////////////////////////////////////////////////////

#ifndef MS_WINDOWS
    #include <sys/mman.h>
#endif

// 返回页大小（字节）
static size_t
get_page_size(void)
{
#ifdef MS_WINDOWS
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
#else
    return sysconf(_SC_PAGESIZE);
#endif
}

// 处理jit相关的错误，并将错误信息格式化为Python异常
static void
jit_error(const char *message)
{
#ifdef MS_WINDOWS
    int hint = GetLastError();
#else
    int hint = errno;
#endif
    PyErr_Format(PyExc_RuntimeWarning, "JIT %s (%d)", message, hint);
}

// 在内存中分配一定大小的空间，并确保该内存是页面对齐的
static unsigned char *
jit_alloc(size_t size)
{
    assert(size);
    assert(size % get_page_size() == 0);
#ifdef MS_WINDOWS
    int flags = MEM_COMMIT | MEM_RESERVE;
    unsigned char *memory = VirtualAlloc(NULL, size, flags, PAGE_READWRITE);
    int failed = memory == NULL;
#else
    int flags = MAP_ANONYMOUS | MAP_PRIVATE;
    unsigned char *memory = mmap(NULL, size, PROT_READ | PROT_WRITE, flags, -1, 0);
    int failed = memory == MAP_FAILED;
#endif
    if (failed) {
        jit_error("unable to allocate memory");
        return NULL;
    }
    return memory;
}

// 当jit优化失败后，需要将为其分配的空间释放
static int
jit_free(unsigned char *memory, size_t size)
{
    assert(size);
    assert(size % get_page_size() == 0);
#ifdef MS_WINDOWS
    int failed = !VirtualFree(memory, 0, MEM_RELEASE);
#else
    int failed = munmap(memory, size);
#endif
    if (failed) {
        jit_error("unable to free memory");
        return -1;
    }
    return 0;
}

static int
mark_executable(unsigned char *memory, size_t size) // 标记为可执行（失败返回非0）
{
    if (size == 0) {
        return 0;
    }
    assert(size % get_page_size() == 0);
    // Do NOT ever leave the memory writable! Also, don't forget to flush the
    // i-cache (I cannot begin to tell you how horrible that is to debug):
#ifdef MS_WINDOWS
    if (!FlushInstructionCache(GetCurrentProcess(), memory, size)) {
        jit_error("unable to flush instruction cache");
        return -1;
    }
    int old;
    int failed = !VirtualProtect(memory, size, PAGE_EXECUTE_READ, &old);
#else
    __builtin___clear_cache((char *)memory, (char *)memory + size);
    int failed = mprotect(memory, size, PROT_EXEC | PROT_READ);
#endif
    if (failed) {
        jit_error("unable to protect executable memory");
        return -1;
    }
    return 0;
}

// JIT compiler stuff: /////////////////////////////////////////////////////////

// Warning! AArch64 requires you to get your hands dirty. These are your gloves:

// value[value_start : value_start + len]
static uint32_t
get_bits(uint64_t value, uint8_t value_start, uint8_t width)
{
    assert(width <= 32);
    return (value >> value_start) & ((1ULL << width) - 1);
}

// *loc[loc_start : loc_start + width] = value[value_start : value_start + width]
static void
set_bits(uint32_t *loc, uint8_t loc_start, uint64_t value, uint8_t value_start,
         uint8_t width)
{
    assert(loc_start + width <= 32);
    // Clear the bits we're about to patch:
    *loc &= ~(((1ULL << width) - 1) << loc_start);
    assert(get_bits(*loc, loc_start, width) == 0);
    // Patch the bits:
    *loc |= get_bits(value, value_start, width) << loc_start;
    assert(get_bits(*loc, loc_start, width) == get_bits(value, value_start, width));
}

// See https://developer.arm.com/documentation/ddi0602/2023-09/Base-Instructions
// for instruction encodings:
#define IS_AARCH64_ADD_OR_SUB(I) (((I) & 0x11C00000) == 0x11000000)
#define IS_AARCH64_ADRP(I)       (((I) & 0x9F000000) == 0x90000000)
#define IS_AARCH64_BRANCH(I)     (((I) & 0x7C000000) == 0x14000000)
#define IS_AARCH64_LDR_OR_STR(I) (((I) & 0x3B000000) == 0x39000000)
#define IS_AARCH64_MOV(I)        (((I) & 0x9F800000) == 0x92800000)

// LLD is a great reference for performing relocations... just keep in
// mind that Tools/jit/build.py does filtering and preprocessing for us!
// Here's a good place to start for each platform:
// - aarch64-apple-darwin:
//   - https://github.com/llvm/llvm-project/blob/main/lld/MachO/Arch/ARM64.cpp
//   - https://github.com/llvm/llvm-project/blob/main/lld/MachO/Arch/ARM64Common.cpp
//   - https://github.com/llvm/llvm-project/blob/main/lld/MachO/Arch/ARM64Common.h
// - aarch64-pc-windows-msvc:
//   - https://github.com/llvm/llvm-project/blob/main/lld/COFF/Chunks.cpp
// - aarch64-unknown-linux-gnu:
//   - https://github.com/llvm/llvm-project/blob/main/lld/ELF/Arch/AArch64.cpp
// - i686-pc-windows-msvc:
//   - https://github.com/llvm/llvm-project/blob/main/lld/COFF/Chunks.cpp
// - x86_64-apple-darwin:
//   - https://github.com/llvm/llvm-project/blob/main/lld/MachO/Arch/X86_64.cpp
// - x86_64-pc-windows-msvc:
//   - https://github.com/llvm/llvm-project/blob/main/lld/COFF/Chunks.cpp
// - x86_64-unknown-linux-gnu:
//   - https://github.com/llvm/llvm-project/blob/main/lld/ELF/Arch/X86_64.cpp

// Many of these patches are "relaxing", meaning that they can rewrite the
// code they're patching to be more efficient (like turning a 64-bit memory
// load into a 32-bit immediate load). These patches have an "x" in their name.
// Relative patches have an "r" in their name.

// 32-bit absolute address.
void
patch_32(unsigned char *location, uint64_t value)
{
    uint32_t *loc32 = (uint32_t *)location;
    // Check that we're not out of range of 32 unsigned bits:
    assert(value < (1ULL << 32));
    *loc32 = (uint32_t)value;
}

// 32-bit relative address.
void
patch_32r(unsigned char *location, uint64_t value)
{
    uint32_t *loc32 = (uint32_t *)location;
    // 得到offset
    value -= (uintptr_t)location;
    // Check that we're not out of range of 32 signed bits:
    assert((int64_t)value >= -(1LL << 31));
    assert((int64_t)value < (1LL << 31));
    // 存储offset
    *loc32 = (uint32_t)value;
}

// 64-bit absolute address.
void
patch_64(unsigned char *location, uint64_t value)
{
    uint64_t *loc64 = (uint64_t *)location;
    *loc64 = value;
}

// 将value中低12位copy到location所在的位置中
// 12-bit low part of an absolute address. Pairs nicely with patch_aarch64_21r
// (below).
void
patch_aarch64_12(unsigned char *location, uint64_t value)
{
    uint32_t *loc32 = (uint32_t *)location;
    assert(IS_AARCH64_LDR_OR_STR(*loc32) || IS_AARCH64_ADD_OR_SUB(*loc32));
    // There might be an implicit shift encoded in the instruction:
    uint8_t shift = 0;
    if (IS_AARCH64_LDR_OR_STR(*loc32)) {
        shift = (uint8_t)get_bits(*loc32, 30, 2);
        // If both of these are set, the shift is supposed to be 4.
        // That's pretty weird, and it's never actually been observed...
        assert(get_bits(*loc32, 23, 1) == 0 || get_bits(*loc32, 26, 1) == 0);
    }
    value = get_bits(value, 0, 12);
    assert(get_bits(value, 0, shift) == 0);
    set_bits(loc32, 10, value, shift, 12);
}

// 基本作用同上，但是可以提供优化版的patch
// Relaxable 12-bit low part of an absolute address. Pairs nicely with
// patch_aarch64_21rx (below).
void
patch_aarch64_12x(unsigned char *location, uint64_t value)
{
    // This can *only* be relaxed if it occurs immediately before a matching
    // patch_aarch64_21rx. If that happens, the JIT build step will replace both
    // calls with a single call to patch_aarch64_33rx. Otherwise, we end up
    // here, and the instruction is patched normally:
    patch_aarch64_12(location, value);
}

// 将value的低16位copy到location[5:5+16]中，下面的b、c、d也都是copy到5~20位之间，因为在aarch64架构中，一般用5到20位存储立即数
// 16-bit low part of an absolute address.
void
patch_aarch64_16a(unsigned char *location, uint64_t value)
{
    uint32_t *loc32 = (uint32_t *)location;
    assert(IS_AARCH64_MOV(*loc32));
    // Check the implicit shift (this is "part 0 of 3"):
    assert(get_bits(*loc32, 21, 2) == 0);
    set_bits(loc32, 5, value, 0, 16);
}

// 16-bit middle-low part of an absolute address.
void
patch_aarch64_16b(unsigned char *location, uint64_t value)
{
    uint32_t *loc32 = (uint32_t *)location;
    assert(IS_AARCH64_MOV(*loc32));
    // Check the implicit shift (this is "part 1 of 3"):
    assert(get_bits(*loc32, 21, 2) == 1);
    set_bits(loc32, 5, value, 16, 16);
}

// 16-bit middle-high part of an absolute address.
void
patch_aarch64_16c(unsigned char *location, uint64_t value)
{
    uint32_t *loc32 = (uint32_t *)location;
    assert(IS_AARCH64_MOV(*loc32));
    // Check the implicit shift (this is "part 2 of 3"):
    assert(get_bits(*loc32, 21, 2) == 2);
    set_bits(loc32, 5, value, 32, 16);
}

// 16-bit high part of an absolute address.
void
patch_aarch64_16d(unsigned char *location, uint64_t value)
{
    uint32_t *loc32 = (uint32_t *)location;
    assert(IS_AARCH64_MOV(*loc32));
    // Check the implicit shift (this is "part 3 of 3"):
    assert(get_bits(*loc32, 21, 2) == 3);
    set_bits(loc32, 5, value, 48, 16);
}

// 存储21位偏移量的时候，aarch64要求将高2位存储在loc[29:31]，将低19位存储在loc[5:26]中
// B, BL, CBZ, CBNZ, LDR, STR, ADR, ADRP
// 21-bit count of pages between this page and an absolute address's page... I
// know, I know, it's weird. Pairs nicely with patch_aarch64_12 (above).
void
patch_aarch64_21r(unsigned char *location, uint64_t value)
{
    uint32_t *loc32 = (uint32_t *)location;
    value = (value >> 12) - ((uintptr_t)location >> 12);
    // Check that we're not out of range of 21 signed bits:
    assert((int64_t)value >= -(1 << 20));
    assert((int64_t)value < (1 << 20));
    // value[0:2] goes in loc[29:31]:
    set_bits(loc32, 29, value, 0, 2);
    // value[2:21] goes in loc[5:26]:
    set_bits(loc32, 5, value, 2, 19);
}

// Relaxable 21-bit count of pages between this page and an absolute address's
// page. Pairs nicely with patch_aarch64_12x (above).
void
patch_aarch64_21rx(unsigned char *location, uint64_t value)
{
    // This can *only* be relaxed if it occurs immediately before a matching
    // patch_aarch64_12x. If that happens, the JIT build step will replace both
    // calls with a single call to patch_aarch64_33rx. Otherwise, we end up
    // here, and the instruction is patched normally:
    patch_aarch64_21r(location, value);
}

// 28-bit relative branch.
void
patch_aarch64_26r(unsigned char *location, uint64_t value)
{
    uint32_t *loc32 = (uint32_t *)location;
    assert(IS_AARCH64_BRANCH(*loc32));
    value -= (uintptr_t)location;
    // Check that we're not out of range of 28 signed bits:
    assert((int64_t)value >= -(1 << 27));
    assert((int64_t)value < (1 << 27));
    // Since instructions are 4-byte aligned, only use 26 bits:
    assert(get_bits(value, 0, 2) == 0);
    set_bits(loc32, 0, value, 2, 26);
}

// 用于优化和patch AArch64结构中涉及跨页（横跨多个内存页）加载时的指令
// ADRP指令用于计算一个页的基地址，配合LDR指令加载数据或代码，下面的函数目标是优化这两条指令的组合使用，尝试将其转化为更高效的指令组合
// 优化GOT？
// A pair of patch_aarch64_21rx and patch_aarch64_12x.
void
patch_aarch64_33rx(unsigned char *location, uint64_t value)
{
    uint32_t *loc32 = (uint32_t *)location;
    // Try to relax the pair of GOT loads into an immediate value:
    assert(IS_AARCH64_ADRP(*loc32)); // 确保第一条指令时ADRP
    unsigned char reg = get_bits(loc32[0], 0, 5); // 获取第一条指令操作的寄存器
    assert(IS_AARCH64_LDR_OR_STR(loc32[1])); // 检查第二条指令是LDR
    // There should be only one register involved: 确认两条指令使用的是相同的寄存器（操作同一个寄存器）
    assert(reg == get_bits(loc32[1], 0, 5));  // ldr's output register.
    assert(reg == get_bits(loc32[1], 5, 5));  // ldr's input register.
    uint64_t relaxed = *(uint64_t *)value; // 获得绝对地址
    if (relaxed < (1UL << 16)) { // 目标地址小于16位，可以直接作为一个立即数编码在MOVZ指令中，此时ADRP和LDR被优化为MOVZ指令（将16位地址对应的内容加载到寄存器中，并且将寄存器其他位清零）；而LDR被替换为nop
        // adrp reg, AAA; ldr reg, [reg + BBB] -> movz reg, XXX; nop
        loc32[0] = 0xD2800000 | (get_bits(relaxed, 0, 16) << 5) | reg;
        loc32[1] = 0xD503201F;
        return;
    }
    if (relaxed < (1ULL << 32)) {// 如果目标地址小于32位，则ADRP和LDR被优化为MOVZ和MOVK（将立即数左移指定位数并加载到寄存器中，保持寄存器中其他位不变）的组合，也就是先移动16位，再移动16位
        // adrp reg, AAA; ldr reg, [reg + BBB] -> movz reg, XXX; movk reg, YYY
        loc32[0] = 0xD2800000 | (get_bits(relaxed,  0, 16) << 5) | reg;
        loc32[1] = 0xF2A00000 | (get_bits(relaxed, 16, 16) << 5) | reg;
        return;
    }
    relaxed = value - (uintptr_t)location; // 获得相对地址
    if ((relaxed & 0x3) == 0 &&
        (int64_t)relaxed >= -(1L << 19) &&
        (int64_t)relaxed < (1L << 19))
    {// 满足特定的对齐条件并且在19位的范围内，将ADRP和LDR替换为一个LDR和NOP
        // adrp reg, AAA; ldr reg, [reg + BBB] -> ldr reg, XXX; nop
        loc32[0] = 0x58000000 | (get_bits(relaxed, 2, 19) << 5) | reg;
        loc32[1] = 0xD503201F;
        return;
    }
    // Couldn't do it. Just patch the two instructions normally: // 如果都不满足，则执行常规的patch
    patch_aarch64_21rx(location, value); // 将完整的21位页偏移patch到location指示的指令中
    patch_aarch64_12x(location + 4, value); // 将页偏移的低12位patch到location+4指示的指令中
}

// patch 32位相对地址到location指示的指令的对应内容中
// 优化GOT？尝试使得地址可以通过立即数直接获取，而不是再通过全局偏移表来间接访问
// Relaxable 32-bit relative address.
void
patch_x86_64_32rx(unsigned char *location, uint64_t value)
{
    uint8_t *loc8 = (uint8_t *)location;
    // Try to relax the GOT load into an immediate value: 尝试将偏移地址变为立即数
    uint64_t relaxed = *(uint64_t *)(value + 4) - 4;
    if ((int64_t)relaxed - (int64_t)location >= -(1LL << 31) &&
        (int64_t)relaxed - (int64_t)location + 1 < (1LL << 31))
    {
        if (loc8[-2] == 0x8B) {
            // mov reg, dword ptr [rip + AAA] -> lea reg, [rip + XXX]
            loc8[-2] = 0x8D;
            value = relaxed;
        }
        else if (loc8[-2] == 0xFF && loc8[-1] == 0x15) {
            // call qword ptr [rip + AAA] -> nop; call XXX
            loc8[-2] = 0x90;
            loc8[-1] = 0xE8;
            value = relaxed;
        }
        else if (loc8[-2] == 0xFF && loc8[-1] == 0x25) {
            // jmp qword ptr [rip + AAA] -> nop; jmp XXX
            loc8[-2] = 0x90;
            loc8[-1] = 0xE9;
            value = relaxed;
        }
    }
    // 进行实际的patch
    patch_32r(location, value);
}

#include "jit_stencils.h"

// Compiles executor in-place. Don't forget to call _PyJIT_Free later!
int
_PyJIT_Compile(_PyExecutorObject *executor, const _PyUOpInstruction trace[], size_t length)
{
    const StencilGroup *group;
    // Loop once to find the total compiled size:
    uintptr_t instruction_starts[UOP_MAX_TRACE_LENGTH];
    size_t code_size = 0;
    size_t data_size = 0;
    group = &trampoline;
    code_size += group->code_size;
    data_size += group->data_size;
    for (size_t i = 0; i < length; i++) {
        const _PyUOpInstruction *instruction = &trace[i];
        group = &stencil_groups[instruction->opcode];
        instruction_starts[i] = code_size;
        code_size += group->code_size;
        data_size += group->data_size;
    }
    group = &stencil_groups[_FATAL_ERROR];
    code_size += group->code_size;
    data_size += group->data_size;
    // Round up to the nearest page:
    size_t page_size = get_page_size();
    assert((page_size & (page_size - 1)) == 0);
    size_t padding = page_size - ((code_size + data_size) & (page_size - 1));
    size_t total_size = code_size + data_size + padding;
    unsigned char *memory = jit_alloc(total_size); // 分配空间，用于存储JIT编译后的代码
    if (memory == NULL) {
        return -1;
    }
    // Update the offsets of each instruction:
    for (size_t i = 0; i < length; i++) {
        instruction_starts[i] += (uintptr_t)memory;
    }
    // Loop again to emit the code:
    unsigned char *code = memory;
    unsigned char *data = memory + code_size;
    // Compile the trampoline, which handles converting between the native
    // calling convention and the calling convention used by jitted code
    // (which may be different for efficiency reasons). On platforms where
    // we don't change calling conventions, the trampoline is empty and
    // nothing is emitted here:
    group = &trampoline;
    group->emit(code, data, executor, NULL, instruction_starts);
    code += group->code_size;
    data += group->data_size;
    assert(trace[0].opcode == _START_EXECUTOR || trace[0].opcode == _COLD_EXIT);
    // 主循环发射代码，每条指令都生成一个函数，负责copy二进制模板，并将传入的data patch到holes中
    for (size_t i = 0; i < length; i++) {
        const _PyUOpInstruction *instruction = &trace[i];
        group = &stencil_groups[instruction->opcode];
        group->emit(code, data, executor, instruction, instruction_starts);
        code += group->code_size;
        data += group->data_size;
    }
    // Protect against accidental buffer overrun into data:
    group = &stencil_groups[_FATAL_ERROR];
    group->emit(code, data, executor, NULL, instruction_starts);
    code += group->code_size;
    data += group->data_size;
    assert(code == memory + code_size);
    assert(data == memory + code_size + data_size);
    if (mark_executable(memory, total_size)) { // 标记为可执行失败：clear
        jit_free(memory, total_size);
        return -1;
    }
    executor->jit_code = memory; // 将JIT编译后的代码存入executor
    executor->jit_side_entry = memory + trampoline.code_size;
    executor->jit_size = total_size;
    return 0;
}

void
_PyJIT_Free(_PyExecutorObject *executor)
{
    unsigned char *memory = (unsigned char *)executor->jit_code;
    size_t size = executor->jit_size;
    if (memory) {
        executor->jit_code = NULL;
        executor->jit_side_entry = NULL;
        executor->jit_size = 0;
        if (jit_free(memory, size)) {
            PyErr_WriteUnraisable(NULL);
        }
    }
}

#endif  // _Py_JIT
