import requests
from bs4 import BeautifulSoup

# 크롤링할 URL
url = 'https://news.naver.com'

# 웹 페이지 요청
response = requests.get(url)

# 요청이 성공했는지 확인
if response.status_code == 200:
    # BeautifulSoup으로 HTML 파싱
    soup = BeautifulSoup(response.text, 'html.parser')

    # 뉴스 제목 추출 (예: 주요 뉴스 섹션)
    headlines = soup.select('.hdline_article_tit')
    
    print(headlines)

    # 결과 출력
    for index, headline in enumerate(headlines):
        print(f"{index + 1}: {headline.get_text(strip=True)}")
else:
    print("웹 페이지를 가져오는 데 실패했습니다.")