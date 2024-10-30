# 2024-2-CSP: Computational Statistics Project (2024-2)

## 데이터 출처 및 저작권 정보

이 프로젝트는 **NOAA GML**에서 제공한 CO₂ 데이터와 **서울 과거 날씨 데이터(1994-2024)**를 사용하여 수행되었습니다. 해당 데이터는 각각 NOAA GML과 서울 날씨 데이터 제공자의 규정에 따라 공공과 과학 커뮤니티에 무료로 제공되며, 폭넓게 사용되어 더 큰 이해와 새로운 과학적 통찰로 이어질 것을 기대하며 제공됩니다.

### CO₂ 데이터 사용 안내
NOAA GML 데이터의 사용 시, 아래의 저작권 정보를 포함하여 GML의 기여에 대한 공정한 인정을 부탁드립니다. 데이터가 연구 또는 출판물의 중심이 되는 경우, 데이터 제공자에게 연락하여 공동 저자 자격 부여를 고려할 수 있습니다.

#### CO₂ 데이터 제공자 및 문의
- **데이터 제공자**: [Xin Lan (xin.lan@noaa.gov)](mailto:xin.lan@noaa.gov)
- **추가 정보**: [NOAA GML 홈페이지](https://gml.noaa.gov/ccgg/trends/)

> **참고**: 이 데이터는 최신 품질 관리 절차를 거친 예비 데이터일 수 있습니다.
> - 최근 몇 개월간의 데이터는 예비 상태로 제공됩니다.
> - Mauna Loa Observatory가 2022년 화산 폭발로 인해 일시 중단되었으나, 2023년에 관측이 재개되었습니다.

### 서울 과거 날씨 데이터 (1994-2024)

이 데이터셋은 서울의 일일 날씨 조건을 기록한 것으로, 1994년부터 2024년 1월 1일까지의 데이터를 포함합니다. 이 데이터는 매일 또는 이틀마다 갱신되며, 최신 및 과거 날씨 패턴을 분석하는 데 유용합니다.

#### 데이터셋 내용
- **Date**: 기록된 날씨 데이터의 날짜 (datetime 형식)
- **Maximum Temperature (tempmax)**: 해당 일의 최고 기온 (°F)
- **Minimum Temperature (tempmin)**: 해당 일의 최저 기온 (°F)
- **Average Temperature (temp)**: 해당 일의 평균 기온 (°F)
- **Feels Like Temperature (feelslike)**: 습도와 바람을 고려한 체감 온도 (°F)
- **Dew Point (dew)**: 이슬점 온도 (°F)
- **Humidity (humidity)**: 공기 중 습도 비율 (%)
- **Precipitation (precip)**: 총 강수량 (mm)
- **Snow (snow)**: 총 적설량 (mm)
- **Wind Speed (windspeed)**: 평균 풍속 (km/h)
- **Wind Direction (winddir)**: 바람이 불어오는 방향 (도)
- **Sea Level Pressure (sealevelpressure)**: 해수면 대기압 (hPa) ...

#### 데이터셋 라이선스 및 저작권
- **라이선스**: MIT 라이선스
- **추가 정보**: 데이터셋의 사용 시 출처를 명시하고, MIT 라이선스 조건에 따라 자유롭게 활용할 수 있습니다.

이 데이터셋은 **Weather and Climate** 주제로 태그되어 있으며, 분석 및 연구 목적으로 사용될 수 있습니다.
