# Styles Loader

This node uses the same logic as the A111 styles csv.  
The first column is the name, the second the positive, the third the negative.  
A sample [styles.csv](https://github.com/melMass/comfy_mtb/blob/main/styles.csv) gets installed on first run.

> **Note**
> Some styles can have empty columns, for instance I personally use distinct ones for positive and negatives, so be sure to wire the right output.

## Extract Styles
It's sometime useful to be able to directly act on the content of a given style, for that an option was added to the context menu of that node to.. extract the styles to plain text inputs:  
![extract](https://github.com/melMass/comfy_mtb/assets/7041726/c068d770-d5a8-4078-bc3c-20bb6533e42d)
