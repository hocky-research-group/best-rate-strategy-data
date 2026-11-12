# Render ray-traced snapshots of the protein G C-alpha Go model from a LAMMPS DCD.
#
# Usage (headless):
#   vmd -dispdev text -e scripts/render_go_snapshots.tcl -args <psf> <dcd> <outdir> <frame> [<frame> ...]
# <frame> is a 0-based DCD frame index or the word "last", optionally followed by a view
# suffix "@<axis><deg>[<axis><deg>...]" (for example 0@y180 or last@x90y-30) that rotates
# the oriented scene before rendering. One PNG per token is written to
# <outdir>/go_frame<N>[_<view>].png via VMD's bundled Tachyon binary (BMP, converted with sips).
#
# Beads are the 56 C-alpha atoms, bonds the 55 consecutive pairs from the PSF. Color encodes
# the native secondary structure of 1PGB: N-terminal hairpin (beta1 1-8, beta2 13-20), helix
# (23-36), C-terminal hairpin (beta3 42-46, beta4 51-55); loops gray. The folded frame is
# oriented by its inertia tensor and every later frame is aligned to it on the C-alphas, so
# the native hairpins keep their place on the page.

set psf    [lindex $argv 0]
set dcd    [lindex $argv 1]
set outdir [lindex $argv 2]
set frames [lrange $argv 3 end]
set tachyon [file join [file dirname [info nameofexecutable]] tachyon_MACOSXARM64]

mol new $psf type psf waitfor all
mol addfile $dcd type dcd waitfor all
set nf [molinfo top get numframes]
puts "NFRAMES $nf NATOMS [molinfo top get numatoms]"

# colors: Tol palette, same hues as the paper figures
color change rgb 23 0.000 0.447 0.698   ;# blue      N-terminal hairpin
color change rgb 24 0.835 0.369 0.000   ;# vermilion helix
color change rgb 25 0.000 0.620 0.451   ;# green     C-terminal hairpin
color change rgb 26 0.700 0.700 0.700   ;# gray      loops
set all [atomselect top all]
set groups {{"not (resid 1 to 8 or resid 13 to 20 or resid 23 to 36 or resid 42 to 46 or resid 51 to 55)" 26}
            {"resid 1 to 8 or resid 13 to 20" 23}
            {"resid 23 to 36" 24}
            {"resid 42 to 46 or resid 51 to 55" 25}}

display projection Orthographic
display depthcue off
display shadows on
display ambientocclusion on
display aoambient 0.9
display aodirect 0.4
color Display Background white
axes location Off
mol delrep 0 top
foreach g $groups {
    lassign $g seltext cid
    mol representation VDW 0.95 36
    mol color ColorID $cid
    mol selection $seltext
    mol material AOChalky
    mol addrep top
    mol representation Licorice 0.35 36 36
    mol color ColorID $cid
    mol selection $seltext
    mol material AOChalky
    mol addrep top
}
# bonds between groups take the color of the atom at each end: Licorice draws half-bonds,
# so a loop-to-strand bond is gray on one half and blue on the other

# orient the folded frame by its inertia tensor: long axis (smallest moment) along x,
# flattest direction (largest moment) along the viewing axis z; a proper rotation only
$all frame 0
lassign [measure inertia $all -eigenvals] com axes evals
set order [lsort -real -indices $evals]
set ax [lindex $axes [lindex $order 0]]
set az [lindex $axes [lindex $order 2]]
set ay [veccross $az $ax]
set R [list [concat $ax 0] [concat $ay 0] [concat $az 0] {0 0 0 1}]
$all moveby [vecscale -1 $com]
$all move $R
set ref [atomselect top all frame 0]
for {set f 1} {$f < $nf} {incr f} {
    $all frame $f
    $all move [measure fit $all $ref]
}
$all frame 0

foreach tok $frames {
    lassign [split $tok @] fr view
    if {$fr eq "last"} { set fr [expr {$nf - 1}] }
    animate goto $fr
    display resetview
    scale by 1.6
    set tag ""
    foreach {m ax deg} [regexp -all -inline {([xyz])(-?[0-9.]+)} $view] {
        rotate $ax by $deg
        append tag "$ax$deg"
    }
    if {$tag ne ""} { set tag "_$tag" }
    display update
    set base [file join $outdir go_frame$fr$tag]
    render Tachyon $base.dat "\"$tachyon\" -aasamples 12 %s -format BMP -res 2000 1600 -o %s.bmp > /dev/null 2>&1"
    exec sips -s format png $base.dat.bmp --out $base.png > /dev/null
    file delete $base.dat $base.dat.bmp
    puts "WROTE $base.png"
}
quit
