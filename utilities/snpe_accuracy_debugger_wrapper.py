# -*- coding: utf-8 -*-
"""snpe-accuracy-debugger 包装器：绕过 QAIRT SDK 内部的编码 bug。

问题：SDK 的 qti/aisw/accuracy_debugger/lib/device/helpers/nd_device_utilities.py
中 output.decode() 写死 UTF-8，而中文 Windows 下 PowerShell 输出为 GBK/OEM(936)
代码页，decode 抛 UnicodeDecodeError（'utf-8' codec can't decode byte 0xbe...）。

方案：不修改 SDK 文件，运行时把 nd_device_utilities.execute 替换为容错解码
版本（依次尝试 UTF-8 / GBK / UTF-16-LE，最后 errors='replace' 兜底），
再执行原 snpe-accuracy-debugger 脚本。

用法（由 accuracy_debugger.py 调用）：
    python snpe_accuracy_debugger_wrapper.py --inference_engine ...
    原脚本路径由本包装器按平台从 QAIRT_SDK_ROOT 推导。
"""
import os
import sys
import platform
import runpy
import subprocess
from pathlib import Path
from threading import Timer


def _format_output(output):
    """与 SDK 原实现一致的输出格式化。"""
    stripped_out = []
    if output is not None and len(output) > 0:
        stripped_out = [line.strip() for line in output.split('\n') if line.strip()]
    return stripped_out


def _decode_bytes(data):
    """容错解码：依次尝试 UTF-8 / GBK / UTF-16-LE，最后 replace 兜底。"""
    if data is None:
        return ''
    for enc in ('utf-8', 'gbk', 'utf-16-le'):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode('utf-8', errors='replace')


def _safe_execute(command, args=None, cwd='.', shell=False, powershell=False,
                  timeout=3600):
    """替换 nd_device_utilities.execute：逻辑与 SDK 原实现一致，仅解码改为容错。"""
    if args is None:
        args = []
    command_list = [command] + args
    if powershell:
        command_list = ['powershell.exe'] + command_list
    process = subprocess.Popen(command_list, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               cwd=cwd, shell=shell)
    timer = Timer(float(timeout), process.kill)
    timer.start()
    try:
        output, error = process.communicate()
    finally:
        if timer.is_alive():
            timer.cancel()
    return process.returncode, _format_output(_decode_bytes(output)), \
        _format_output(_decode_bytes(error))


def _patch_nd_device_utilities():
    """把 SDK 的 execute 替换为容错版本（模块属性替换，对后续调用方生效）。"""
    from qti.aisw.accuracy_debugger.lib.device.helpers import nd_device_utilities
    nd_device_utilities.execute = _safe_execute
    print("[snpe-accuracy-debugger-wrapper] patched nd_device_utilities.execute "
          "(tolerant decode) OK")


def _resolve_original_script() -> Path:
    """按当前平台推导原 snpe-accuracy-debugger 脚本路径。"""
    sdk_root = os.environ.get('QAIRT_SDK_ROOT')
    if not sdk_root:
        raise SystemExit('[snpe-accuracy-debugger-wrapper] QAIRT_SDK_ROOT not set')
    if sys.platform.startswith('win'):
        machine = platform.machine().lower()
        arch = 'aarch64-windows-msvc' if machine in ('arm64', 'aarch64') else 'x86_64-windows-msvc'
    else:
        arch = 'x86_64-linux-clang'
    script = Path(sdk_root) / 'bin' / arch / 'snpe-accuracy-debugger'
    if not script.exists():
        raise SystemExit(f'[snpe-accuracy-debugger-wrapper] script not found: {script}')
    return script


def main():
    script = _resolve_original_script()
    _patch_nd_device_utilities()
    # 原脚本从 sys.argv[1:] 解析参数；runpy 以 __main__ 执行其模块级 main()
    sys.argv = [str(script)] + sys.argv[1:]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
