; =============================================================================
;   macros.inc: Standard macros for program structure.
; =============================================================================

%macro program_exit 0
    mov ah, 0x4C
    int 0x21
%endmacro

%macro program_begin 0
    org 0x100
    section .text
    global _start
_start:
%endmacro

%macro program_end 0
    program_exit
%endmacro