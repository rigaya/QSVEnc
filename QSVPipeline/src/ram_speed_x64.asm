
section .code
    align 16

section .text

global read_sse

;void __stdcall read_sse(uint8_t *src, uint32_t size, uint32_t count_n) (
;  [rcx] PIXEL_YC       *src
;  [rdx] uint32_t        size
;  [r8]  uint32_t        count_n
;)

read_sse:
        push rdi
        push rsi
        push rbx

        mov edi, 128
        mov rsi, r8
        mov r9,  rcx
        mov eax, edx
        shr eax, 7
        align 16
    .OUTER_LOOP:
        mov rbx, r9
        mov rdx, rbx
        add rdx, 64
        mov ecx, eax
    .INNER_LOOP:
        movaps xmm0, [rbx];
        movaps xmm1, [rbx+16];
        movaps xmm2, [rbx+32];
        movaps xmm3, [rbx+48];
        add rbx, rdi;
        movaps xmm4, [rdx];
        movaps xmm5, [rdx+16];
        movaps xmm6, [rdx+32];
        movaps xmm7, [rdx+48];
        add rdx, rdi
        dec ecx
        jnz .INNER_LOOP

        dec esi
        jnz .OUTER_LOOP
        
        pop rbx
        pop rsi
        pop rdi

        ret



global read_avx

;void __stdcall read_avx(uint8_t *src, uint32_t size, uint32_t count_n) (
;  [rcx] PIXEL_YC       *src
;  [rdx] uint32_t        size
;  [r8]  uint32_t        count_n
;)

read_avx:
        push rdi
        push rsi
        push rbx

        mov edi, 256
        mov rsi, r8
        mov r9,  rcx
        mov eax, edx
        shr eax, 8
        align 16
    .OUTER_LOOP:
        mov rbx, r9
        mov rdx, rbx
        add rdx, 128
        mov ecx, eax
    .INNER_LOOP:
        movaps xmm0, [rbx];
        movaps xmm1, [rbx+32];
        movaps xmm2, [rbx+64];
        movaps xmm3, [rbx+96];
        add rbx, rdi
        movaps xmm4, [rdx];
        movaps xmm5, [rdx+32];
        movaps xmm6, [rdx+64];
        movaps xmm7, [rdx+96];
        add rdx, rdi
        dec ecx
        jnz .INNER_LOOP

        dec esi
        jnz .OUTER_LOOP

        vzeroupper
        
        pop rbx
        pop rsi
        pop rdi

        ret

global write_sse

;void __stdcall _write_sse(uint8_t *src, uint32_t size, uint32_t count_n) (
;  [esp+08] PIXEL_YC       *src
;  [esp+16] uint32_t        size
;  [esp+20] uint32_t        count_n
;)

write_sse:
        push rdi
        push rsi
        push rbx

        mov edi, 128
        mov rsi, r8
        mov r9,  rcx
        mov eax, edx
        shr eax, 7
        align 16
    .OUTER_LOOP:
        mov rbx, r9
        mov rdx, rbx
        add rdx, 64
        mov ecx, eax
    .INNER_LOOP:
        movaps [rbx],    xmm0 
        movaps [rbx+16], xmm0 
        movaps [rbx+32], xmm0
        movaps [rbx+48], xmm0
        add rbx, rdi
        movaps [rdx],    xmm0 
        movaps [rdx+16], xmm0 
        movaps [rdx+32], xmm0
        movaps [rdx+48], xmm0
        add rdx, rdi
        dec ecx
        jnz .INNER_LOOP

        dec esi
        jnz .OUTER_LOOP
        
        pop rbx
        pop rsi
        pop rdi

        ret




global write_avx

;void __stdcall _write_avx(uint8_t *src, uint32_t size, uint32_t count_n) (
;  [esp+08] PIXEL_YC       *src
;  [esp+16] uint32_t        size
;  [esp+20] uint32_t        count_n
;)

write_avx:
        push rdi
        push rsi
        push rbx

        mov edi, 256
        mov rsi, r8
        mov r9,  rcx
        mov eax, edx
        shr eax, 8
        align 16
    .OUTER_LOOP:
        mov rbx, r9
        mov rdx, rbx
        add rdx, 128
        mov ecx, eax
    .INNER_LOOP:
        vmovaps [rbx],    ymm0 
        vmovaps [rbx+32], ymm0 
        vmovaps [rbx+64], ymm0
        vmovaps [rbx+96], ymm0
        add rbx, rdi
        vmovaps [rdx],    ymm0 
        vmovaps [rdx+32], ymm0 
        vmovaps [rdx+64], ymm0
        vmovaps [rdx+96], ymm0
        add rdx, rdi
        dec ecx
        jnz .INNER_LOOP

        dec esi
        jnz .OUTER_LOOP

        vzeroupper
        
        pop rbx
        pop rsi
        pop rdi

        ret
