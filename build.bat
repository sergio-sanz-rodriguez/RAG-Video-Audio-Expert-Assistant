@echo off
echo Cleaning up old build...
rmdir /s /q build
rmdir /s /q dist
del /q /f app.spec

REM Run pyinstaller with your environment's Python
"D:\Repos\ML_Projects\RAG-Video-Audio-Expert-Assistant\env\python.exe" -m PyInstaller --onefile --clean ^
--collect-all gradio ^
--collect-all safehttpx ^
--collect-all groovy ^
--collect-submodules transformers.models ^
--collect-submodules langchain.chains.conversational_retrieval ^
--collect-submodules chromadb ^
--hidden-import=langchain.chains.conversational_retrieval.base ^
--hidden-import=chromadb.telemetry.product.posthog ^
--icon=images/rocket_icon.ico ^
--add-data "vector_db_1024;vector_db_1024" ^
--add-data "modules;modules" ^
--add-data ".\env\Lib\site-packages\gradio_client\types.json;gradio_client" ^
app.py

pause
