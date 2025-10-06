; =============================================================================
;   io.inc: Standard I/O procedures for 16-bit NASM.
; =============================================================================

section .text

; --- Prints a null-terminated string ---
; Input: SI = pointer to string
PrintString:
    push ax
.loop:
    mov al, [si]
    cmp al, 0
    je .done
    mov ah, 0x0E        ; BIOS teletype output
    int 0x10
    inc si
    jmp .loop
.done:
    pop ax
    ret

; --- Prints the 16-bit unsigned value in AX as a decimal ---
PrintU16:
    push bx
    push cx
    push dx

    cmp ax, 0
    jne .start_division
    mov al, '0'
    mov ah, 0x0E
    int 0x10
    jmp .done_printing

.start_division:
    mov cx, 0
    mov bx, 10
.divide_loop:
    xor dx, dx
    div bx
    push dx
    inc cx
    cmp ax, 0
    jne .divide_loop

.print_loop:
    pop dx
    add dl, '0'
    mov ah, 0x0E
    mov al, dl
    int 0x10
    loop .print_loop

.done_printing:
    pop dx
    pop cx
    pop bx
    ret