@echo on
call C:\Users\qcrew\Anaconda3\Scripts\activate.bat
call conda activate qcrew-dev
cd qcrew/control/stage
python -m stage_teardown
