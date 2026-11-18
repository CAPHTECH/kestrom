# Which meter a question about a building asks for, and whether its floor is low or high: two heads, answered apart.
in ac: match "空調|エアコン|冷房|暖房"
in light: match "照明|ライト"
in not_ac: match "(?:空調|エアコン)(?:じゃなく|ではなく|以外)"   # "not the air conditioning"
in floors: count "\\d+階"                                        # how many floors the question names
in floor: number "(\\d+)階"                                      # the first of them
h1 a: all(ac, !not_ac)
h1 l: any(light)
h1 one: 1*floors >= 1
h1 many: 1*floors >= 2
h1 high: 1*floor >= 5
h2 AC: all(a, !l)
h2 Light: all(l, !a)
h2 Low: all(one, !many, !high)
h2 High: all(one, !many, high)
out meter/ac: 1*AC
out meter/light: 1*Light
out floor/low: 1*Low
out floor/high: 1*High
