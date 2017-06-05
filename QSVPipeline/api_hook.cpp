// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------


#include "api_hook.h"

void apihook::print_all_func() {
    hook_function(nullptr, nullptr, nullptr);
}

void *apihook::hook_function(const TCHAR *pTargetModuleName, const char *pTargetFunctionName, void *pNewFunc) {
    for (int i = 0; i < 2; i++) {
        if (void *pModuleBase = GetModuleHandle((i == 0 && pTargetModuleName) ? pTargetModuleName : NULL)) {
            ULONG nSize = 0;
            for (auto pImageImportDescriptor = (PIMAGE_IMPORT_DESCRIPTOR)ImageDirectoryEntryToData((HMODULE)pModuleBase, TRUE, IMAGE_DIRECTORY_ENTRY_IMPORT, &nSize);
                pImageImportDescriptor->Name; pImageImportDescriptor++) {
                const char *pModuleName = ((const char*)pModuleBase + pImageImportDescriptor->Name);
                for (PIMAGE_THUNK_DATA
                    pFirstThunk    = (PIMAGE_THUNK_DATA)((char*)pModuleBase + pImageImportDescriptor->FirstThunk),
                    pOrgFirstThunk = (PIMAGE_THUNK_DATA)((char*)pModuleBase + pImageImportDescriptor->OriginalFirstThunk);
                    pFirstThunk->u1.Function;
                    pFirstThunk++, pOrgFirstThunk++) {
                    if (!IMAGE_SNAP_BY_ORDINAL(pOrgFirstThunk->u1.Ordinal)) {
                        PIMAGE_IMPORT_BY_NAME pImportName = (PIMAGE_IMPORT_BY_NAME)((char*)pModuleBase + (size_t)pOrgFirstThunk->u1.AddressOfData);
                        if (pNewFunc) {
                            if (_stricmp((const char*)pImportName->Name, pTargetFunctionName) == 0) {
                                DWORD dwOldProtect;
                                if (!VirtualProtect(&pFirstThunk->u1.Function, sizeof(pFirstThunk->u1.Function), PAGE_READWRITE, &dwOldProtect))
                                    return nullptr;

                                void *pOrigFunc = (void*)pFirstThunk->u1.Function;
                                WriteProcessMemory(GetCurrentProcess(), &pFirstThunk->u1.Function, &pNewFunc, sizeof(pFirstThunk->u1.Function), NULL);
                                pFirstThunk->u1.Function = (size_t)pNewFunc;

                                VirtualProtect(&pFirstThunk->u1.Function, sizeof(pFirstThunk->u1.Function), dwOldProtect, &dwOldProtect);
                                return pOrigFunc;
                            }
                        } else {
                            printf("Module:%s Hint:%d, Name:%s\n", pModuleName, pImportName->Hint, pImportName->Name);
                        }
                    }
                }
            }
        }
    }
    return nullptr;
}

apihook::apihook() : hookList() {

}

apihook::~apihook() {
    fin();
}

int apihook::hook(const TCHAR *pTargetModuleName, const char *pTargetFunctionName, void *pNewFunc, void **pOrigFuncSetAdr) {
    hookData hook;
    hook.sTargetModuleName = pTargetModuleName;
    hook.sTargetFunctionName = pTargetFunctionName;
    hook.pNewFunc = pNewFunc;
    hook.pOrigFunc = hook_function(hook.sTargetModuleName.c_str(), hook.sTargetFunctionName.c_str(), hook.pNewFunc);
    if (!hook.pOrigFunc) {
        return 1;
    }
    if (pOrigFuncSetAdr) {
        hook.pOrigFuncSetAdr = pOrigFuncSetAdr;
        *hook.pOrigFuncSetAdr = hook.pOrigFunc;
    }
    hookList.push_back(hook);
    return 0;
}

void apihook::fin() {
    for (const auto& hook : hookList) {
        if (hook.pOrigFunc) {
            hook_function(hook.sTargetModuleName.c_str(), hook.sTargetFunctionName.c_str(), hook.pOrigFunc);
            if (hook.pOrigFuncSetAdr) {
                *hook.pOrigFuncSetAdr = nullptr;
            }
        }
    }
    hookList.clear();
}
void apihook::fin(const TCHAR *pTargetModuleName, const char *pTargetFunctionName) {
    auto hook = std::find_if(hookList.begin(), hookList.end(), [pTargetModuleName, pTargetFunctionName](hookData data) {
        return data.sTargetModuleName == pTargetModuleName && data.sTargetFunctionName == pTargetFunctionName;
    });
    if (hook != hookList.end()) {
        if (hook->pOrigFunc) {
            hook_function(hook->sTargetModuleName.c_str(), hook->sTargetFunctionName.c_str(), hook->pOrigFunc);
            if (hook->pOrigFuncSetAdr) {
                *(hook->pOrigFuncSetAdr) = nullptr;
            }
            hookList.erase(hook);
        }
    }
}
void apihook::fin(void *pOrigFuc) {
    if (pOrigFuc) {
        auto hook = std::find_if(hookList.begin(), hookList.end(), [pOrigFuc](hookData data) {
            return data.pOrigFunc == pOrigFuc;
        });
        if (hook != hookList.end()) {
            hook_function(hook->sTargetModuleName.c_str(), hook->sTargetFunctionName.c_str(), hook->pOrigFunc);
            if (hook->pOrigFuncSetAdr) {
                *(hook->pOrigFuncSetAdr) = nullptr;
            }
            hookList.erase(hook);
        }
    }
}
void *apihook::get_orig(const TCHAR *pTargetModuleName, const char *pTargetFunctionName) {
    auto hook = std::find_if(hookList.begin(), hookList.end(), [pTargetModuleName, pTargetFunctionName](hookData data) {
        return data.sTargetModuleName == pTargetModuleName && data.sTargetFunctionName == pTargetFunctionName;
    });
    return (hook != hookList.end()) ? hook->pOrigFunc : nullptr;
}
