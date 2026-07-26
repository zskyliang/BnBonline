# ImageGen 首轮提示词与版本记录

- 生成日期：2026-07-26
- 版本：pilot-v1
- 生成器：Codex 内置 ImageGen
- 风格参考：`../source/animal-style-reference.png`
- 用途：原创 Blender 建模参考，不直接作为游戏贴图
- 状态：已于 2026-07-26 通过验收门 1

## 八角色风格母版

- 文件：`../pilot/animal-roster-style-master.png`
- 尺寸：1698 × 926
- SHA-256：`7e1fc31b8fed69384d365cc7a758131e79904b8ee72b0462514b5ef0d53f11ed`

```text
Use case: stylized-concept
Asset type: game character style master for Blender modeling
Input images: Image 1 is art-direction reference only; create original designs, do not copy its exact characters or layout.
Primary request: Design a coherent lineup of exactly eight original upright dumb-cute forest animal game characters: cat, dog, rabbit, brown bear, fox, raccoon, penguin, and capybara.
Scene/backdrop: clean warm off-white handmade paper, no environment.
Style/medium: hand-drawn children's picture-book character concept art; confident charcoal ink contour; subtle watercolor washes and visible paper grain; low-saturation warm palette.
Character language: head-to-body ratio about 1:1, huge round eyes occupying much of the face, tiny pupils with slightly unfocused/misaligned gaze, tiny mouth, round belly, very short thick legs, stubby arms held slightly away from body, upright bipedal posture, stiff and lovably clumsy personality.
Composition/framing: landscape model sheet; one full-body front three-quarter pose per species, evenly spaced in one row or two balanced rows; every character fully visible at matching scale with clear silhouette.
Team-color design: preserve natural species fur colors; reserve clearly visible local accent patches on ear tips, paws, tail bands, flipper edges, or back patches for runtime recoloring; include a very thin colored foot ring under each character without obscuring feet.
Constraints: exactly eight distinct species; no costumes, jobs, weapons, props, UI, labels, captions, letters, logos, or watermark; no duplicate species; no realistic anatomy; no anime style; no glossy 3D render; keep eyes, belly, muzzle, and species markings outside the recolor zones.
```

## 猫四视图

- 文件：`../pilot/cat-turnaround.png`
- 尺寸：1693 × 929
- SHA-256：`e71417b7400e38e788eeeb799335993b04c885e588ed3a4dff57be264939c4f5`

```text
Use case: stylized-concept
Asset type: Blender character turnaround reference sheet
Input images: Image 1 is the original hand-drawn art-direction reference; Image 2 is the approved working style master. Preserve the exact gray cat design language from Image 2 while creating a new clean turnaround.
Primary request: Create one internally consistent upright gray tabby cat model sheet showing exactly four full-body views of the same character: front orthographic, right-side orthographic, back orthographic, and front-right three-quarter view.
Scene/backdrop: warm off-white handmade paper, completely uncluttered.
Style/medium: charcoal ink contour with subtle watercolor wash and visible paper grain; original children's picture-book game concept art.
Character proportions: head and body each about half total height; huge round white eyes, very small dark pupils with a slightly unfocused mismatched gaze, tiny pink nose and mouth, round belly, very short thick legs, stubby arms held about 25 degrees away from body, upright neutral modeling pose.
Species details: triangular ears, three subtle forehead tabby stripes, white muzzle and belly, long curled tail with alternating gray and pale team-tint placeholder bands.
Runtime recolor zones: clearly and consistently show pale desaturated blue placeholder color only on ear tips, forepaws, hind paws, selected tail bands, and a small upper-back patch; all other fur, belly, muzzle, eyes, nose, and tabby markings remain fixed.
Composition/framing: landscape sheet, four views evenly spaced, same baseline and exact scale, full body and tail visible in every view, minimal perspective distortion.
Constraints: exactly one character identity across all views; no action poses, labels, text, arrows, dimension lines, props, clothes, scenery, ground shadows, logos, or watermark; do not crop ears, paws, or tail; do not add extra limbs; no glossy 3D rendering.
```

## 猫动作表

- 文件：`../pilot/cat-animation-poses.png`
- 尺寸：1690 × 931
- SHA-256：`4c951c7734a4f973dc418a4846d2e3ed01813a7fb4b5060ca59299aeb78c6350`

```text
Use case: stylized-concept
Asset type: Blender animation pose reference sheet
Input images: Image 1 is the eight-character style master; Image 2 is the exact gray cat turnaround and is the identity/proportion source of truth.
Primary request: Draw exactly six separated full-body poses of the same gray tabby cat from Image 2 for a shared 8 FPS game animation set.
Poses, left to right: neutral idle with a tiny vacant sway; stiff dumb-cute waddle with one short leg forward and opposite arm lagging; placing a round soap bubble with a shallow awkward squat; trapped inside a large squishy oval bubble with stubby limbs splayed; harmless comedic defeat flopped sideways with dazed eyes; victory with one tiny two-foot hop and arms raised.
Scene/backdrop: clean warm off-white handmade paper.
Style/medium: charcoal ink contour, subtle watercolor wash, paper grain, children's picture-book animation key poses.
Character invariants: preserve the exact gray fur, huge eyes, tiny pupils, tabby stripes, white belly and muzzle, short legs, curled striped tail, pale-blue local recolor zones, and 1:1 head/body ratio from Image 2.
Motion language: intentionally rigid and clumsy, low-energy, cute, readable silhouette; no athletic motion and no extreme squash.
Composition/framing: landscape contact sheet, six evenly spaced poses at matching scale; character and tail fully visible; bubble only in the placing and trapped poses.
Constraints: no labels, text, arrows, frame numbers, props other than the specified soap bubble, scenery, UI, logos, watermark, extra limbs, costume changes, identity drift, anime expressions, pain, injury, or violence.
```

## 熊四视图

- 文件：`../pilot/bear-turnaround.png`
- 尺寸：1693 × 929
- SHA-256：`61d8d906a5639412611984e1cf4dd176311ae4f023b9e6747908f45c5636e79e`

```text
Use case: stylized-concept
Asset type: Blender character turnaround reference sheet
Input images: Image 1 is the original hand-drawn art-direction reference; Image 2 is the approved working style master. Preserve the exact brown bear design language from Image 2 while creating a new clean turnaround.
Primary request: Create one internally consistent upright round brown bear model sheet showing exactly four full-body views of the same character: front orthographic, right-side orthographic, back orthographic, and front-right three-quarter view.
Scene/backdrop: warm off-white handmade paper, completely uncluttered.
Style/medium: charcoal ink contour with subtle watercolor wash and visible paper grain; original children's picture-book game concept art.
Character proportions: head and body each about half total height; huge round white eyes, very small dark pupils with a slightly unfocused mismatched gaze, tiny muzzle and mouth, very large round belly, very short thick legs, stubby arms held about 25 degrees away from body, upright neutral modeling pose.
Species details: small round ears, warm medium-brown fur, lighter tan muzzle and belly, tiny round tail, broad but soft silhouette.
Runtime recolor zones: clearly and consistently show pale desaturated violet placeholder color only on ear centers, forepaws, hind paws, and a small upper-back oval patch; all other fur, belly, muzzle, eyes, nose, and species markings remain fixed.
Composition/framing: landscape sheet, four views evenly spaced, same baseline and exact scale, full body and tail visible in every view, minimal perspective distortion.
Constraints: exactly one character identity across all views; no action poses, labels, text, arrows, dimension lines, props, clothes, scenery, ground shadows, logos, or watermark; do not crop ears or paws; do not add extra limbs; no glossy 3D rendering.
```

## 熊动作表

- 文件：`../pilot/bear-animation-poses.png`
- 尺寸：1691 × 930
- SHA-256：`62878bde3d594a14bdb475739b0dfca948b69cfe569011b2eba59d834fd8c85a`

```text
Use case: stylized-concept
Asset type: Blender animation pose reference sheet
Input images: Image 1 is the eight-character style master; Image 2 is the exact brown bear turnaround and is the identity/proportion source of truth.
Primary request: Draw exactly six separated full-body poses of the same round brown bear from Image 2 for a shared 8 FPS game animation set.
Poses, left to right: neutral idle with a tiny vacant sway; stiff dumb-cute waddle with one short leg forward and opposite arm lagging; placing a round soap bubble with a shallow awkward squat; trapped inside a large squishy oval bubble with stubby limbs splayed; harmless comedic defeat flopped sideways with dazed eyes; victory with one tiny two-foot hop and arms raised.
Scene/backdrop: clean warm off-white handmade paper.
Style/medium: charcoal ink contour, subtle watercolor wash, visible paper grain, children's picture-book animation key poses.
Character invariants: preserve the exact warm brown fur, huge round eyes, tiny slightly mismatched pupils, lighter tan muzzle and belly, small round ears, very short thick legs, tiny round tail, pale-violet local recolor zones, and 1:1 head/body ratio from Image 2.
Motion language: intentionally rigid and clumsy, heavy low-energy weight shifts, cute, readable silhouette; no athletic motion and no extreme squash.
Composition/framing: landscape contact sheet, six evenly spaced poses at matching scale; character fully visible; bubble only in the placing and trapped poses.
Constraints: no labels, text, arrows, frame numbers, props other than the specified soap bubble, scenery, UI, logos, watermark, extra limbs, costume changes, identity drift, anime expressions, pain, injury, or violence.
```
