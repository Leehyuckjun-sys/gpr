# YOLO(px→m) → UWB 보정(GPR) 파이프라인

이 레포는 **YOLO의 바닥 픽셀 좌표를 호모그래피로 m 단위 (x_ip, y_ip)** 로 변환한 뒤,  
**UWB 좌표(uwb_x, uwb_y)** 를 기준으로 **Gaussian Process Regression**으로 오차(dx, dy)를 학습하고  
온라인에서 **보정된 좌표(x_corr, y_corr)** 를 출력합니다.

## 요구사항
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
