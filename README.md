# survival-metrics
Various implementations of survival metrics, especially Antolini's C-index.

BUILD:
flit build

TODO:
- try to be flexible in how outcomes are passed: two arrays, one structured (with arbitrary names?)
- assume a certain api for models in sklearn-style scorers?
- write a tutorial notebook
- recover tests for comparing different implementations
- ipcw?
- handle competing? not a priority