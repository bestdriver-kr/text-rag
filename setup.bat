@echo off
chcp 65001 >nul
echo ============================================
echo   Ollama RAG 설치 스크립트
echo ============================================
echo.

:: Python 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않습니다.
    echo.
    echo 아래 주소에서 Python 3.11 이상을 설치하세요:
    echo https://www.python.org/downloads/
    echo.
    echo 설치 시 "Add Python to PATH" 체크박스를 반드시 선택하세요!
    pause
    exit /b 1
)

echo [1/4] Python 확인됨
python --version

:: pip 업그레이드
echo.
echo [2/4] pip 업그레이드 중...
python -m pip install --upgrade pip

:: 패키지 설치
echo.
echo [3/4] 필요한 패키지 설치 중...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [오류] 패키지 설치 실패
    pause
    exit /b 1
)

:: Ollama 모델 다운로드
echo.
echo [4/4] Ollama 모델 다운로드 중...
echo.
echo 임베딩 모델 (nomic-embed-text) 다운로드...
ollama pull nomic-embed-text
echo.
echo LLM 모델 (gemma3:4b) 다운로드... (약 2.5GB, 시간이 걸릴 수 있습니다)
ollama pull gemma3:4b

echo.
echo ============================================
echo   설치 완료!
echo ============================================
echo.
echo 사용법:
echo   run.bat index --folder "C:\내문서폴더"   ^<-- 문서 인덱싱
echo   run.bat chat                              ^<-- 대화 모드
echo   run.bat ask "질문 내용"                   ^<-- 단일 질문
echo   run.bat list                              ^<-- 인덱싱된 파일 목록
echo.
pause
