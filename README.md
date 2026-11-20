# fluency game site

Hosted copy of the signature fluency rating game. Live at https://moonfellow.pythonanywhere.com.

Generated, not edited here. Source of truth: `fluency_game.py` in the signatures repo.
Rebuilt by `hosted_game.py deploy` there, which also refreshes this folder.

- `fluency_wsgi.py`: WSGI entry PythonAnywhere runs
- `pool.csv`: frozen 500-crop pool, anon ids, hashed groups, pairing traits
- `crops/`: crop images named by anon id
- `fluency/`: rating CSVs, server only, not tracked (rater names)

No salt, anon map or genealogy names here.
