"""Configure the reference hand locally before first launch."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import grip
if __name__ == '__main__':
    grip.prepare()
    ref={k:float(input(label+' (mm): ')) for k,label in [('hand_length','Reference hand length'),('hand_breadth','Reference hand breadth'),('palm_length','Reference palm length')]}
    print('This calibration uses the supplied grip orientation: thumb end A, palm angle 50 degrees, region 10–90%.')
    print('Inspect data/preparation.json for the repair report before accepting.')
    if input('Accept repaired mesh and supplied landmarks? Type yes: ').strip().lower()!='yes':
        raise SystemExit('Calibration not saved.')
    cal=grip.calibration(dict(reference=ref,thumb_end=-1,palm_angle=50,usable_start=.1,usable_end=.9,accepted_repairs=True,landmarks_confirmed=True,source_sha256=grip.digest(grip.SOURCE)))
    path=grip.ROOT/'data/calibration.json'
    if path.exists():
        raise SystemExit('Calibration already exists; back it up and remove it explicitly to replace it.')
    path.write_text(json.dumps(cal,indent=2))
    print('Saved local calibration. Run app.py next.')
