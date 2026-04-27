import sys
import os

# Fix Windows console encoding for emoji
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from app import create_app

app = create_app()

if __name__ == '__main__':
    print()
    print("=" * 50)
    print("  FruitFresh Antigravity Predictor")
    print("  Open: http://localhost:5000")
    print("=" * 50)
    print()
    app.run(debug=True, host='0.0.0.0', port=5000)
