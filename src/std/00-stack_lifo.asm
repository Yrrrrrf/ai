; =============================================================================
;   std/00-stack_lifo.asm
;   Demonstrates the LIFO nature of the stack using the standard library.
; =============================================================================

; --- Include our standard library ---
%include "../lib/macros.inc"
%include "../lib/io.inc"

; --- Begin the program using our macro ---
program_begin

    ; --- Step 1: Load registers with different values ---
    mov ax, 1111
    mov bx, 9999

    ; --- Step 2: Push them onto the stack in order ---
    push ax ; 1111 is now on the stack
    push bx ; 9999 is on top of 1111

    ; --- Step 3: Clear registers to prove pop works ---
    xor ax, ax
    xor bx, bx

    ; --- Step 4: Pop them back in LIFO order ---
    pop ax  ; AX now holds 9999 (Last-In)
    pop bx  ; BX now holds 1111 (First-Out)

    ; --- Step 5: Print the results to show they swapped ---
    call PrintU16   ; Prints AX (9999)
    call PrintNewline
    
    mov ax, bx      ; Move BX to AX for printing
    call PrintU16   ; Prints original BX value (1111)
    call PrintNewline

; --- End the program using our macro ---
program_end

; --- Data for this specific program ---
section .data
    ; The newline is now handled by io.inc, so no data section is needed
    ; unless we had other unique messages.