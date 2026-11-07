# chess-set-template

A template for making your own 3D-printable chess set. Drop a logo SVG into `assets/`, run one command, and you get a full set of print-ready STL files with your emblem on every piece, a preview image and a printability check.

It is the generator behind [chess-set-ai](https://github.com/GIAMAT13/chess-set-ai), with the finished models and brand logos taken out.

## Quick start

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
# put your logo in assets/, e.g. assets/mylogo.svg
python src/build_all.py mylogo
```

You get:

- `stl/mylogo/`: one STL per piece and `plate_mylogo_16_pieces.stl` (a whole side on one plate)
- `images/preview_mylogo.png`: a render of the six pieces
- a printability report in the terminal (written for a Bambu Lab P1S; see below)

`python src/build_all.py` with no names builds every SVG in `assets/`, plus two built-in emblems: `plain` (a small network graph) and `star` (a starburst). `assets/example.svg` (a crown) is there to try it out; delete it when you add your own.

## Make it yours

### 1. The emblem (easy)

Add `assets/<name>.svg`. The file name becomes the version name. The SVG reader:

- handles `path`, `circle`, `ellipse`, `rect` and `polygon`
- drops an app-icon background square, blurred glow layers and gradients
- cuts white shapes out of coloured ones
- joins a logo made of separate parts with thin bridges, so each piece stays one part
- puts logos that stand on top of a piece (bishop, queen, king) on a round coin, so thin or hanging parts cannot break off

Best results: a simple, filled logo with no thin lines (under about 1 mm at print size they come out as single lines). Transforms (`transform="..."`) are not applied, so flatten them first (in Inkscape: Path > Object to Path, then save as plain SVG).

### 2. The pieces (Python)

Each piece is one function in `src/build_set.py`, built with [manifold3d](https://github.com/elalish/manifold) (CSG: add and subtract simple solids):

| Function | Piece | Current theme |
|---|---|---|
| `pawn()` | Pawn | a neuron with dendrites |
| `rook()` | Rook | a GPU tower with fans |
| `knight()` | Knight | a horse head with plug-prong ears and a cable mane (`KNIGHT_PTS` is its side outline) |
| `bishop()` | Bishop | a terminal window with a `>_` prompt |
| `queen()` | Queen | a context window sliding up a ribbon of tokens |
| `king()` | King | a small neural network |
| `chip_base()` | every base | a microchip with pins |

Helpers at the top of the file (`cyl`, `sphere`, `box`, `tube`, `revolve`, `rounded_box`, `ycyl`, `emblem_xz`, `medallion`, `on_disc`) cover most shapes. A revolved profile (`revolve([(r, z), ...])`) is the quickest way to make a classic piece body. Where the emblem goes on each piece is the `emblem_xz(...)` / `medallion(...)` call in that function.

### 3. Check that it prints

```bash
python src/print_check.py mylogo
```

For every STL it checks:

- it fits the build volume (`BED`)
- it has enough bed contact
- it does not tip over easily
- nothing starts in mid-air (slice by slice, as the slicer sees it)
- there are no thin necks that would snap
- walls are at least 2 lines of a 0.4 mm nozzle

It ends with `ALL OK` or a list of what to fix. For another printer, change `BED` and `NOZZLE` at the top of the file.

## All commands

```bash
python src/build_all.py [names...]      # STLs, preview and print check, per version
python src/build_set.py <name>          # only the STLs -> stl/<name>/
python src/render_preview.py <name>     # only the preview -> images/preview_<name>.png
python src/print_check.py [names...]    # only the print check
python src/check_parts.py <name>        # count the separate parts of each piece
python src/store_images.py <name>       # banner, cover and piece images -> store/images/
```

## Sharing on MakerWorld or Thingiverse

`store/listing.md` is an example listing (title, description, print settings, tags, licence, AI disclosure and a checklist). Change it to match your set, and run `src/store_images.py` for the images. Do not upload sets with logos you do not own: a trademark complaint can take the model down.

## Printing

- Scale 100%: the king is about 86 mm tall, bases are 27 to 36 mm wide, and a 16-piece plate is 167 x 164 mm.
- PLA or PETG, 0.16 or 0.2 mm layers, 3 walls, 15% gyroid infill, no supports.
- You need 1 king, 1 queen, 2 bishops, 2 knights, 2 rooks and 8 pawns per side. Print one side in white and one in black.

## License

MIT, see [LICENSE](LICENSE). Logos you add are not covered by it.
