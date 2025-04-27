“When to use ;”
- Use it to end:

- Variable declarations: int x = 5;

- Expression statements: x++;

- typedef/using: using Vec = std::vector<int>;

- Forward declarations: class Foo;

- Full class/struct/enum definitions: class Bar { … };

Don’t use it:

- Right after function bodies

- Directly after if(...), for(...), while(...), switch(...)

- After closing a namespace (or extern-"C") block

- With preprocessor directives