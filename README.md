# fsu_grid_gen

Multiblock curvilinear grids for wall-resolved LES of the FSU Mach 3 mixed-compression
intake–isolator in OpenSBLI/OPS. Faceted baseline, with a backward-facing-step notch variant in progress.

The intake is built from its defining physics and fixed points (MoC isentropic turn, oblique shocks,
specified corners), validated against CAD, and only then meshed. Geometry and meshing are
separate steps: the mesher never constructs geometry, it only reads the validated solution.

## Pipeline

```
1_geometry.py   CFG → solve → validate → export        → solved/geometry.json, walls.npz, figs/
2_mesh.py       solved/ → x, y, z distributions → blocks → checks → HDF5   → grids/<case>/
```

```bash
./1_geometry.py                        # must report 0 FAIL(s)
./2_mesh.py                            # dry run: prints block sizes, checks, figures
./2_mesh.py --nz 50 --write            # write the OPS grid
```

`2_mesh.py` refuses geometry that failed validation or whose `walls.npz` hash doesn't match
`geometry.json` (override with `--allow-unvalidated`). Rerun step 1 after any CFG change.

## Layout

```
1_geometry.py          geometry driver: CFG, REF, TOL, validation, export, figures
2_mesh.py              mesh driver: MESH config, distributions, blocks, checks, writer call
geometry/              geometry library
  gasdynamics.py         Prandtl–Meyer, θ-β-M, oblique shock
  lines.py               intersect, ang
  moc.py                 focused_isentropic_wall (simple-wave compression through a focus)
  geometry.py            solve_geometry: all corners and wall polylines from CFG
meshing/
  ops_writer.py          streaming OPS HDF5 writer (k-slab at a time, linear halos)
data/                  inputs: CAD spline samples, reference images
solved/                step-1 output (gitignored)
grids/<case>/          step-2 output: .h5, params.json, figs/ (gitignored)
zz_archive/            retired generators and converters (not used)
```

Imports run one way only: `gasdynamics, lines → moc → geometry → 1_geometry.py`.
Library code never imports a driver.

## Geometry (step 1)

All lengths are CAD mm. Inputs (`CFG`):

| key | value | source |
|---|---|---|
| `Minf` | 3.0 | spec |
| `theta0`, `theta1` | 6.67°, 20.8° | CAD ramp angles |
| `cowl_angle`, `facet_angle` | 9°, 9° | spec |
| `leading_edge` | (0, 0.0992) | ramp tangent point on the r = 0.1 LE radius |
| `cowl_lip` | (140.4, 50.0) | spec |
| `focus` | (140.696, 49.696) | MoC focus, least-squares fit to CAD (0.43 from the lip) |
| `shoulder_x` | 156.1754 | CAD spline end |
| `ceiling`, `floor` | 56.44, 39.10 | sketch (H = 17.34) |
| `iso_length` | 90.0 | from facet end. **Assumption: check the FSU paper** |

Construction: LE shock (θ-β-M) → straight ramp at θ0 → MoC turn θ0→θ1 with every
compression wave through the focus → straight at θ1 to the shoulder → facet at 9° to the floor →
flat isolator. Cowl: 9° from the lip to the ceiling, then flat.

Validation (`figs/validation.txt`) checks every segment angle, the isolator height, MoC wave focusing,
the wall against the CAD spline (max |dy| 0.029), and corners against digitised points.

Derived corners (current):

| point | x | y |
|---|---|---|
| turn start | 63.6425 | 7.5417 |
| turn end | 114.3373 | 18.9386 |
| shoulder | 156.1754 | 34.8314 |
| cowl end | 181.0606 | 56.44 |
| facet end | 183.1264 | 39.10 |
| outlet | 273.1264 | 39.10 / 56.44 |

## Mesh (step 2)

Faceted topology, 4 blocks, split at the cowl lip:

```
 [3][   2   ]     3/2: external flow above the lip line / cowl (coarse, not studied)
 [0][   1   ]     0/1: ramp → facet → isolator
```

- **x:** piecewise uniform (dx ≈ 0.1026) with a node exactly on every corner.
- **y:** Vinokur two-sided stretching per column, with end spacings prescribed:
  - 0.006 at every wall (ramp, facet, floor, both cowl faces)
  - ~0.1 on the lip line upstream
  - 0.30 at the far field
  - the outer cowl relaxes to 0.15 within 5 mm of the lip
- **z:** `periodic` (uniform, endpoint excluded) or `sidewall` (wall layer dz 0.006 plus a uniform core,
  switched on from the lip, blended in over 20 mm upstream).

Main settings live in `MESH` in `2_mesh.py`. CLI overrides: `--dx --ny-lower --ny-upper --dy-wall --nz
--span --write`.

Current faceted grid: nx 2663, ny 500 + 150.

| block | np0 (x) | np1 (y) |
|---|---|---|
| 0 | 1369 | 500 |
| 1 | 1295 | 500 |
| 2 | 1295 | 150 |
| 3 | 1369 | 150 |

That's 1.73M points per z-plane: 86.6M at nz = 50, 346M at nz = 200.

Checks printed every run:
- node mismatch on all interfaces (must be 0)
- spacing ratio across flow interfaces (≈1)
- y growth (lower ≤ 1.02, upper ≤ 1.05)
- minimum wall cell angle (69°)
- readback of the written file

## Output format

`grids/<case>/<case>.h5`, matching OpenSBLI's expected layout:

- `/opensbliblockNN/x0_BN, x1_BN, x2_BN, h0_BN`, float64, shape `(nz+2h, ny+2h, nx+2h)`, 5 halos
- linear-extrapolated halos in all directions, including edges and corners
- coordinates divided by `scale_factor` = 17.34 × 0.07 / 1.584 = 0.76629 mm per δ*
- root attribute `provenance_json` plus `params.json` record the mesh config and geometry hash

## OpenSBLI settings

- `blockNnp0/np1/np2`: printed by `2_mesh.py`
- `Deltas = 1.0`: the curvilinear metrics cancel them
- `Lz = 25 / scale_factor` (= 32.625): the spanwise period used by the wall forcing
- `Lx0 = x_out / scale_factor` (= 356.4): flow-through reporting only

On LUMI, stripe the grid directory before writing (`lfs setstripe -c 8 grids/<case>`). Pass the OPS
decomposition as CLI arguments (`OPS_PROCESSES_PER_BLOCK=...`, `OPS_FORCE_DECOMP_X=...`), not as
environment variables.

## Variants

| variant | status |
|---|---|
| faceted | done |
| notched | in progress |

The notch is a 1 mm vertical step at the shoulder, with the floor at 10.6° until it meets the
isolator floor at x ≈ 184.33. It will be selected with `2_mesh.py --variant notched` and adds a 6th block
(the layer on the notched floor).

## Known caveats

- `iso_length` = 90 is an assumption.
- `scale_factor` uses H = 17.34. Earlier production grids used 17.3.
- The smallest cells (wall corners, sidewall corners, notch step) set the CFL limit. Refining
  `dy_wall` or adding sidewalls reduces dt.
