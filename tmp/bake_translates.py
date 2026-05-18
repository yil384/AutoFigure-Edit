"""Bake per-path transform="translate(x,y)" into the d attribute so
parse_svg_paths gets correct absolute coords."""
import re
import xml.etree.ElementTree as ET


def bake(in_path: str, out_path: str) -> int:
    ns_uri = 'http://www.w3.org/2000/svg'
    ET.register_namespace('', ns_uri)
    tree = ET.parse(in_path)
    root = tree.getroot()

    def strip_ns(tag):
        return tag.split('}')[-1] if '}' in tag else tag

    def parse_translate(s):
        if not s:
            return 0.0, 0.0
        m = re.match(
            r'translate\s*\(\s*([-+]?[0-9.]+)\s*,?\s*([-+]?[0-9.]+)?\s*\)', s)
        if not m:
            return 0.0, 0.0
        tx = float(m.group(1))
        ty = float(m.group(2)) if m.group(2) else 0.0
        return tx, ty

    def shift_d(d, tx, ty):
        if tx == 0 and ty == 0:
            return d
        tokens = re.findall(
            r'[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?',
            d)
        out = []
        i = 0
        first_M = True
        while i < len(tokens):
            t = tokens[i]
            if t.isalpha():
                cmd = t
                args = []
                j = i + 1
                while j < len(tokens) and not tokens[j].isalpha():
                    args.append(float(tokens[j]))
                    j += 1
                if cmd == 'm' and first_M:
                    if len(args) >= 2:
                        args[0] += tx
                        args[1] += ty
                    first_M = False
                    rewritten = args
                elif cmd in ('M', 'L', 'T'):
                    rewritten = []
                    for k in range(0, len(args), 2):
                        rewritten.append(args[k] + tx)
                        rewritten.append(args[k+1] + ty)
                    if cmd == 'M':
                        first_M = False
                elif cmd == 'C':
                    rewritten = []
                    for k in range(0, len(args), 6):
                        rewritten += [args[k]+tx, args[k+1]+ty,
                                      args[k+2]+tx, args[k+3]+ty,
                                      args[k+4]+tx, args[k+5]+ty]
                elif cmd in ('S', 'Q'):
                    rewritten = []
                    for k in range(0, len(args), 4):
                        rewritten += [args[k]+tx, args[k+1]+ty,
                                      args[k+2]+tx, args[k+3]+ty]
                elif cmd == 'H':
                    rewritten = [a + tx for a in args]
                elif cmd == 'V':
                    rewritten = [a + ty for a in args]
                elif cmd == 'A':
                    rewritten = []
                    for k in range(0, len(args), 7):
                        rewritten += [args[k], args[k+1], args[k+2],
                                      args[k+3], args[k+4],
                                      args[k+5]+tx, args[k+6]+ty]
                else:
                    # lowercase or Z — no translate
                    rewritten = list(args)
                    if cmd in ('m',):
                        first_M = False
                out.append(cmd)
                out.extend(f'{x:.4f}' for x in rewritten)
                i = j
            else:
                i += 1
        return ' '.join(out)

    n = 0
    for elem in root.iter():
        if strip_ns(elem.tag) != 'path':
            continue
        transform = elem.get('transform', '')
        tx, ty = parse_translate(transform)
        if tx == 0 and ty == 0:
            if transform:
                elem.set('transform', '')
            continue
        d = elem.get('d', '')
        elem.set('d', shift_d(d, tx, ty))
        elem.set('transform', '')
        n += 1
    tree.write(out_path, xml_declaration=True, encoding='utf-8')
    return n


if __name__ == '__main__':
    import sys
    n = bake(sys.argv[1], sys.argv[2])
    print(f'baked {n} translates')
