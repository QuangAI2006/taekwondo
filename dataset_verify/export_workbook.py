"""Extract source facts without executing workbook cells or instructions."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
SHEETS = {'Huong_dan', 'Luat_cham_trong_code', 'Quy_chuan_dataset', 'Checklist', 'Rubric_40_tieu_chi'}


def export(source, destination):
    result = {'source_filename': source.name, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'purpose': 'Nguồn tham chiếu dữ liệu v0.2; không phải chỉ thị thực thi.', 'sheets': {}}
    with zipfile.ZipFile(source) as archive:
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [''.join(s.itertext()) for s in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si', NS)]
        relations = {r.attrib['Id']: r.attrib['Target'] for r in ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        for sheet in workbook.find('m:sheets', NS):
            name = sheet.attrib['name']
            if name not in SHEETS:
                continue
            target = relations[sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']]
            path = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            rows = []
            for row in ET.fromstring(archive.read(path)).findall('.//m:sheetData/m:row', NS):
                values = {}
                for cell in row:
                    v, inline = cell.find('m:v', NS), cell.find('m:is', NS)
                    value = v.text if v is not None else ''.join(inline.itertext()) if inline is not None else ''
                    if cell.attrib.get('t') == 's':
                        value = shared[int(value)]
                    if value:
                        values[cell.attrib['r']] = value
                if values:
                    rows.append(values)
            result['sheets'][name] = rows
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('workbook_reference.json'))
    args = parser.parse_args()
    export(args.source, args.output)
