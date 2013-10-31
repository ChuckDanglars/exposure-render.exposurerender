set wd = %CD%

echo "Starting exposure render compositor"
cd compositor
start compositor.exe

sleep 1
cd..

echo "Starting exposure render renderer"
cd renderer
start renderer.exe

sleep 1
cd..

echo "Starting exposure render gui"
cd gui
start gui.exe