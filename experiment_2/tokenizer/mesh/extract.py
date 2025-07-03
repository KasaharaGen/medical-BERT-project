import xml.etree.ElementTree as ET

mesh_path = "desc2025.xml"  # NLMから取得したMeSH XMLファイル
tree = ET.parse(mesh_path)
root = tree.getroot()

target_prefixes = ["D006973", "C01"]  # D006973: Dengue, C01: Infection
selected_terms = []

for record in root.findall("DescriptorRecord"):
    treenums = record.findall("TreeNumberList/TreeNumber")
    if any(t.text.startswith(tuple(target_prefixes)) for t in treenums):
        name = record.find("DescriptorName/String")
        if name is not None:
            selected_terms.append(name.text.strip().lower())

# 重複除去して保存
selected_terms = sorted(set(selected_terms))
with open("mesh_dengue_infection_terms.txt", "w") as f:
    for term in selected_terms:
        f.write(term + "\n")

print(f"✅ 抽出語数: {len(selected_terms)}")
