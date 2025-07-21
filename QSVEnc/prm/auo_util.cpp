#include "rgy_osdep.h"
#include "auo_util.h"

static inline BOOL is_space_or_crlf(int c) {
    return (c == ' ' || c == '\r' || c == '\n');
}

BOOL del_arg(char *cmd, char *target_arg, int del_arg_delta) {
    char *p_start, *ptr;
    char * const cmd_fin = cmd + strlen(cmd);
    del_arg_delta = clamp(del_arg_delta, -1, 1);
    //指定された文字列を検索
    if ((p_start = strstr(cmd, target_arg)) == NULL)
        return FALSE;
    //指定された文字列の含まれる部分の先頭を検索
    for ( ; cmd < p_start; p_start--)
        if (is_space_or_crlf(*(p_start-1)))
            break;
    //指定された文字列の含まれる部分の最後尾を検索
    ptr = p_start;
    {
        BOOL dQB = FALSE;
        while (is_space_or_crlf(*ptr))
            ptr++;

        while (cmd < ptr && ptr < cmd_fin) {
            if (*ptr == '"') dQB = !dQB;
            if (!dQB && is_space_or_crlf(*ptr))
                break;
            ptr++;
        }
    }
    if (del_arg_delta < 0)
        std::swap(p_start, ptr);

    //次の値を検索
    if (del_arg_delta) {
        while (cmd <= ptr + del_arg_delta && ptr + del_arg_delta < cmd_fin) {
            ptr += del_arg_delta;
            if (!is_space_or_crlf(*ptr)) {
                break;
            }
        }

        BOOL dQB = FALSE;
        while (cmd < ptr && ptr < cmd_fin) {
            if (*ptr == '"') dQB = !dQB;
            if (!dQB && is_space_or_crlf(*ptr))
                break;
            ptr += del_arg_delta;
        }
    }
    //文字列の移動
    if (del_arg_delta < 0)
        std::swap(p_start, ptr);

    memmove(p_start, ptr, (cmd_fin - ptr + 1) * sizeof(cmd[0]));
    return TRUE;
}

