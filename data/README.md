# Data

The files the notebook reads are not committed. When the terms let you republish them, they are assets on the GitHub release tagged `data`. The release is public. When the terms let you download the files but not republish them, do not create that release. Write the command that downloads them from the source, and say what it needs (an account, a key in `.env`, a path).

`OWNER/REPO` is your fork, the form `username/aii600-final-project`. In the clone, once:

```bash
gh repo set-default OWNER/REPO
```

Create the release from the project root, after the files are in this folder:

```bash
gh release create data data/your-file.csv --repo OWNER/REPO --title "Data" --notes "See data/README.md"
```

Add every file the notebook reads. When a file changes:

```bash
gh release upload data data/your-file.csv --repo OWNER/REPO --clobber
```

On another machine:

```bash
gh release download data --dir data --repo OWNER/REPO
```

Record a SHA-256 for each file. Replacing a file with `--clobber` does not show up in git, so update the hash in the same commit:

```bash
shasum -a 256 data/your-file.csv
```

Replace the lines below.

- Delft Systematic Deadrise Series, republishable
- 5-Regression Data.xlsx
- E1C5F68605A515BED79937F126D8A7405EF1A62A1A45CAB2EC8BD926DC5F20E1
- Individual resistance runs, steady-state (time-averaged), 3325 runs (357 hull configurations)
- License below

Dataset: Delft Systematic Deadrise Series
Creator(s): TU Delft
Copyright: © 1963-2013 TU Delft
License: CC BY 4.0
License link: [\[Creative Commons Attribution 4.0 International license\]](https://creativecommons.org/licenses/by/4.0/)
Citations:
    1. Keuning, L., & Hillege, W. (2017). The results of the Delft Systematic Deadrise Series. In Proceedings of the 14th International Conference on Fast Sea Transportation (FAST 2017): Innovative Materials (pp. 97–106). FAST Organizing Committee.
    2. Clement, E. P., & Blount, D. L. (1963). Resistance tests of a systematic series of planing hull forms. SNAME Transactions, 71, 491–579.
Changes: No changes made.

After processing:
- SHA: F583E0BD7BC509CADCBC78FF3E7912390570E04564B3D396F4F2E96561EFDFC5