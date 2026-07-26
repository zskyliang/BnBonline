# Audio asset provenance

The shipped mix uses a soft, cute toy-and-bubble palette. The previous
electronic battle loop and crunchy sci-fi explosion are no longer included.
Only the source files listed below are shipped.

## Background music

- Shipped file: `music/battle_loop.ogg`
- Original: `levelmusicloop-tigrun.ogg`
- Title: “Two Simple Game Music Loops”
- Author: qubodup
- Source: <https://opengameart.org/content/two-simple-game-music-loops>
- License: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)
- Downloaded: 2026-07-26
- Processing: copied without transcoding; Godot enables seamless forward looping

The source is explicitly tagged “cute”, “hopeful”, “simple” and “seamless”.

## Water-bubble cues

- Shipped files: `sfx/appear.wav`, `sfx/lay.wav`
- Originals: `bubbles-single2.wav`, `bubbles-single1.wav`
- Title: “Bubble Sound Effects”
- Author: BMacZero (Brian MacIntosh)
- Source: <https://opengameart.org/content/bubble-sound-effects>
- License: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)
- Downloaded: 2026-07-26
- Processing: copied without waveform modification

## Explosion pop

- Shipped file: `sfx/explode.wav`
- Original: `pop5.wav`
- Title: “Pop sounds”
- Author: EZduzziteh
- Source: <https://opengameart.org/content/pop-sounds-0>
- License: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)
- Downloaded: 2026-07-26
- Processing: copied without waveform modification; playback is attenuated by
  an additional 8 dB and receives a subtle upward pitch variation

The selected source is a short vocal bubble pop tagged “casual” and “cute”,
without the mechanical crunch and low-frequency impact of the previous cue.

## Pizzicato and toy-like stingers

`start`, `save`, `die` and `draw` use Kenney's
[Music Jingles](https://kenney.nl/assets/music-jingles) pack. `get` uses
Kenney's [Interface Sounds](https://kenney.nl/assets/interface-sounds) pack.
Both official asset pages identify their license as CC0; Kenney's
[license guidance](https://kenney.nl/support) confirms commercial use is
permitted and attribution is optional.

The victory cue comes from OpenGameArt:

- Shipped file: `sfx/win.ogg`
- Original: `WinPizzicato.ogg` in `winjingle.zip`
- Title: “Win Jingle”
- Author: Fupi
- Source: <https://opengameart.org/content/win-jingle>
- License: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)

Selected cue mapping:

| Shipped file | Original file | Source | Game event |
| --- | --- | --- | --- |
| `sfx/start.ogg` | `jingles_PIZZI04.ogg` | Kenney Music Jingles | round start |
| `sfx/get.ogg` | `pluck_001.ogg` | Kenney Interface Sounds | item pickup |
| `sfx/save.ogg` | `jingles_PIZZI08.ogg` | Kenney Music Jingles | confirmed action |
| `sfx/die.ogg` | `jingles_PIZZI01.ogg` | Kenney Music Jingles | actor defeated |
| `sfx/win.ogg` | `WinPizzicato.ogg` | OpenGameArt Win Jingle | stage victory |
| `sfx/draw.ogg` | `jingles_PIZZI03.ogg` | Kenney Music Jingles | draw or loss |

All listed OGG files are copied without transcoding. `SHA256SUMS` records the
exact shipped subset.
