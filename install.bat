@echo off
echo Installing Email Spam Detection System Dependencies...
echo.

REM Update pip first
python -m pip install --upgrade pip

REM Install packages individually to avoid conflicts
python -m pip install streamlit
python -m pip install pandas
python -m pip install numpy
python -m pip install scikit-learn
python -m pip install matplotlib
python -m pip install seaborn
python -m pip install wordcloud

echo.
echo Installation complete!
echo.
echo To run the application, use:
echo streamlit run app.py
echo.
pause