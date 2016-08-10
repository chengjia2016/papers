# Used for rename downloaded file from Firefoxls plugin: DownThemAll

for file in *._html_; do
    mv "$file" "`basename "$file" ._html_`.html"
done
