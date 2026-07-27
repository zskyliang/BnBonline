# Audio asset provenance

The shipped mix uses a soft, cute toy-and-bubble palette. The previous
electronic battle loop and crunchy sci-fi explosion are no longer included.
Only the source files listed below are shipped.

## Background music

- Shipped file: `music/puddle_jumpers_loop.ogg`
- Original: `Puddle_Jumpers.mp3`
- Provenance: supplied directly by the project owner
- Received: 2026-07-27
- Processing: decoded to 44.1 kHz stereo PCM, moved the loop boundary 1.5
  seconds into the song, crossfaded the original ending into the opening over
  1.4 seconds, and encoded to Ogg Vorbis. Godot also enables forward looping.

The processed loop is 57.5 seconds long. Rights and distribution permission
for the user-provided original remain the responsibility of the project owner.

## Gentle rain and distant thunder ambience

- Shipped file: `ambience/gentle_rain_thunder_loop.ogg`
- Original: `Light Rain Distant Thunder July 5th 2016.wav`
- Author: kvgarlic
- Source:
  <https://commons.wikimedia.org/wiki/File:Light_Rain_Distant_Thunder_July_5th_2016.wav>
- Original Freesound source:
  <https://freesound.org/people/kvgarlic/sounds/349454/>
- License:
  [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)
- Downloaded: 2026-07-26
- Processing: selected a calm forest passage, high-pass filtered at 70 Hz,
  low-pass filtered at 12 kHz, crossfaded the boundary for looping, and encoded
  to stereo Ogg Vorbis. Runtime playback is mixed at `-19 dB` beneath the
  battle music.

The Commons description identifies this as a peaceful Midwest forest rain
recording with distant thunder. It is used only while a match is active.

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
