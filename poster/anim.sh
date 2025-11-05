# compile the LaTeX document that creates the pdf frames
pdflatex anim.tex

# convert them to png
magick -density 600 anim.pdf frame_%03d.png

# find the maximum width and height among all frames
identify -format "%w %h\n" frame_*.png | awk '{if($1>W)W=$1; if($2>H)H=$2} END{print W "x" H}'

# pad the frames (HxW extracted from previous command)
magick frame_*.png -background white -gravity northwest -extent 3289x1435 padded_%03d.png

# create the animated gif
magick -dispose previous -delay 300 -loop 0 padded_*.png anim.gif

# clean up
rm frame_*.png
rm padded_*.png
