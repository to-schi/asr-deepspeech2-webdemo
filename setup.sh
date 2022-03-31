mkdir -p ~/.streamlit/echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
[theme]\n\
primaryColor = "#d0fcff"\n\
backgroundColor = "#00172B"\n\
secondaryBackgroundColor = "#0083B8"\n\
textColor = "#FFF"\n\
font = "sans serif"\n\
\n\
" > ~/.streamlit/config.toml