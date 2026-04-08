

echo ""
echo "=============================="
echo "Summary"
echo "=============================="
python3 -c "
import glob
import json

rows = []
for f in glob.glob('outputs/eval/*.jsonl'):
    lines = [json.loads(l) for l in open(f)]
    format_correct = sum(l['format_reward'] == 1.0 for l in lines)
    answer_correct = sum(l['answer_reward'] == 1.0 for l in lines)
    total = len(lines)
    name = f.split('/')[-1].replace('.jsonl', '')
    rows.append((
        answer_correct / total,
        format_correct / total,
        name,
        format_correct,
        answer_correct,
        total,
    ))

for answer_acc, format_acc, name, format_correct, answer_correct, total in sorted(rows, reverse=True):
    print(
        f'  {name:<45} '
        f'answer={answer_acc:.4f}\t'
        f'format={format_acc:.4f}'
    )
"
