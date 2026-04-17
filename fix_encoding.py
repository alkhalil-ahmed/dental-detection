import os

# Mojibake patterns: UTF-8 chars that look like latin-1 decoded UTF-8 special chars
# Each tuple is (corrupted_string, clean_replacement)
FIXES = [
    # em dash — (U+2014) → encoded as â€" in mojibake
    ('\u00e2\u20ac\u201d', '&mdash;'),
    # non-breaking hyphen / en-dash variants
    ('\u00e2\u20ac\u2019', '-'),
    # left single quote '
    ('\u00e2\u20ac\u2018', "'"),
    # left double quote "
    ('\u00e2\u20ac\u201c', '"'),
    # ellipsis …
    ('\u00e2\u20ac\u00a6', '...'),
    # en dash –
    ('\u00e2\u20ac\u2013', '-'),
    # middle dot · (Â·)
    ('\u00c2\u00b7', '·'),
    # non-breaking space (Â )
    ('\u00c2\u00a0', ' '),
    # right double quote " (â€)
    ('\u00e2\u20ac', '"'),
]

root_dir = 'templates'
fixed_files = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    for fname in filenames:
        if not fname.endswith('.html'):
            continue
        fpath = os.path.join(dirpath, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            original = f.read()
        content = original
        for bad, good in FIXES:
            content = content.replace(bad, good)
        if content != original:
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(content)
            fixed_files.append(fpath)

if fixed_files:
    for p in fixed_files:
        print('Fixed:', p)
else:
    print('No mojibake found in templates')

print('Done.')
