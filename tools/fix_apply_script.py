#!/usr/bin/env python3
from pathlib import Path

path = Path(__file__).with_name("apply_final_hardening.py")
text = path.read_text(encoding="utf-8")
old = '''    ''' + "'''" + '''        "n runtime_mean_ms runtime_median_ms runtime_q25_ms runtime_q75_ms "
        "runtime_err_low_ms runtime_err_high_ms mean_error l1_error\\n"
''' + "'''" + ''',
    ''' + "'''" + '''        "n runtime_mean_ms runtime_median_ms runtime_q25_ms runtime_q75_ms "
        "runtime_err_low_ms runtime_err_high_ms mean_error mean_error_q25 "
        "mean_error_q75 mean_error_err_low mean_error_err_high l1_error l1_error_q25 "
        "l1_error_q75 l1_error_err_low l1_error_err_high\\n"
''' + "'''" + ''',
'''
# The checked-in script accidentally used interpreted newlines in these source
# matching literals. Replace that exact block with raw-string literals.
start = text.index('replace(\n    test_plot,\n    \'\'\'        "n runtime_mean_ms')
end = text.index('\n)\nreplace(\n    test_plot,', start) + 3
replacement = r'''replace(
    test_plot,
    r''' + "'''" + r'''        "n runtime_mean_ms runtime_median_ms runtime_q25_ms runtime_q75_ms "
        "runtime_err_low_ms runtime_err_high_ms mean_error l1_error\n"
''' + "'''" + r''',
    r''' + "'''" + r'''        "n runtime_mean_ms runtime_median_ms runtime_q25_ms runtime_q75_ms "
        "runtime_err_low_ms runtime_err_high_ms mean_error mean_error_q25 "
        "mean_error_q75 mean_error_err_low mean_error_err_high l1_error l1_error_q25 "
        "l1_error_q75 l1_error_err_low l1_error_err_high\n"
''' + "'''" + r''',
)'''
path.write_text(text[:start] + replacement + text[end:], encoding="utf-8", newline="\n")
Path(__file__).unlink()
