set wd = %CD%

cd compositor
start compositor.exe

cd %wd%
cd renderer
start renderer.exe

cd %wd%
cd gui
start gui.exe