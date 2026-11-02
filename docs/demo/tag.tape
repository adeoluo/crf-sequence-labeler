# Records docs/demo/tag.gif from a real run: `vhs docs/demo/tag.tape` from the repo root.
Output docs/demo/tag.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 440
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate && clear` Enter
Sleep 10s
Show
Type `python -m crf_sequence_labeler tag Alice lives in Paris` Enter
Sleep 2.5s
Type `python -m crf_sequence_labeler evaluate` Enter
Sleep 3s
