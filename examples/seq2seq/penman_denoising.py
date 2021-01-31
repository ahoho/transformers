AMR_SOURCES = [
    "(w / want-01 :ARG0 (b / boy) :ARG1 (g / go-01 :ARG0 b))"
]

AMR_TARGETS = [
    "The boy wants to go."
]

AMR_NOISED = {
 "convert-to-triples": {
  "src": "order Graph: ( want :ARG0 ( boy ) :ARG1 ( go :ARG0 boy ) )",
  "tgt": "<t> want :ARG0 boy <t> want :ARG1 go <t> go :ARG0 boy"
 },
 "generate-from-triples": {
  "src": "order Graph: <t> want :ARG0 boy <t> want :ARG1 go <t> go :ARG0 boy",
  "tgt": "The boy wants to go."
 },
 "mask-all": {
  "src": "denoise Graph: ( want :ARG0 <extra_id_0> boy ) :ARG1 ( go :ARG0 boy ) )",
  "tgt": "<extra_id_0> ( <extra_id_1>"
 },
 "mask-all-drop": {
  "src": "denoise Graph: ( want :ARG0  boy ) :ARG1 ( go :ARG0 boy ) )",
  "tgt": " ( "
 },
 "mask-all-mass": {
  "src": "denoise Graph: ( want :ARG0 <extra_id_0> boy ) :ARG1 ( go :ARG0 boy ) )",
  "tgt": "( want :ARG0 ( boy ) :ARG1 ( go :ARG0 boy ) )"
 },
 "mask-components": {
  "src": "denoise Graph: ( want :ARG0 <extra_id_0> boy ) :ARG1 ( go :ARG0 boy ) )",
  "tgt": "<extra_id_0> ( <extra_id_1>"
 },
 "mask-components-corrupt": {
  "src": "denoise Graph: ( want :ARG0 ( boy ) :ARG1 ( go :ARG0 boy ) )",
  "tgt": "( want :ARG0 ( boy ) :ARG1 ( go :ARG0 boy ) )"
 },
 "mask-components-drop": {
  "src": "denoise Graph: ( want :ARG0  boy ) :ARG1 ( go :ARG0 boy ) )",
  "tgt": " ( "
 },
 "mask-components-mass": {
  "src": "denoise Graph: ( want :ARG0 <extra_id_0> boy ) :ARG1 ( go :ARG0 boy ) )",
  "tgt": "( want :ARG0 ( boy ) :ARG1 ( go :ARG0 boy ) )"
 },
 "mask-nodes": {
  "src": "denoise Graph: ( <extra_id_0> :ARG0 ( boy ) :ARG1 ( go :ARG0 <extra_id_1> ) )",
  "tgt": "<extra_id_0> want <extra_id_1> boy <extra_id_2>"
  },
 "mask-nodes-drop": {
  "src": "denoise Graph: (  :ARG0 ( boy ) :ARG1 ( go :ARG0  ) )",
  "tgt": " want  boy "
 },
 "mask-nodes-mass": {
  "src": "denoise Graph: ( <extra_id_0> :ARG0 ( boy ) :ARG1 ( go :ARG0 <extra_id_0> ) )",
  "tgt": "( want :ARG0 ( boy ) :ARG1 ( go :ARG0 boy ) )"
 },
 "mask-surface": {
  "src": "denoise Graph: The boy wants <extra_id_0> go.",
  "tgt": "<extra_id_0> to <extra_id_1>"
 },
 "parse-from-triples": {
  "src": "order Graph: <t> want :ARG0 boy <t> want :ARG1 go <t> go :ARG0 boy",
  "tgt": "( want :ARG0 ( boy ) :ARG1 ( go :ARG0 boy ) )"
 },
 "randomize": {
  "src": "( go :ARG1-of ( want :ARG0 ( boy ) ) :ARG0 boy )",
  "tgt": "The boy wants to go."
 },
 "randomize_convert-to-triples": {
  "src": "order Graph: ( go :ARG1-of ( want :ARG0 ( boy ) ) :ARG0 boy )",
  "tgt": "<t> want :ARG1 go <t> want :ARG0 boy <t> go :ARG0 boy"
 },
 "randomize_generate-from-triples": {
  "src": "order Graph: <t> want :ARG1 go <t> want :ARG0 boy <t> go :ARG0 boy",
  "tgt": "The boy wants to go."
 },
 "randomize_mask-all": {
  "src": "denoise Graph: ( go :ARG1-of <extra_id_0> want :ARG0 ( boy ) ) :ARG0 boy )",
  "tgt": "<extra_id_0> ( <extra_id_1>"
 },
 "randomize_mask-all-drop": {
  "src": "denoise Graph: ( go :ARG1-of  want :ARG0 ( boy ) ) :ARG0 boy )",
  "tgt": " ( "
 },
 "randomize_mask-all-mass": {
  "src": "denoise Graph: ( go :ARG1-of <extra_id_0> want :ARG0 ( boy ) ) :ARG0 boy )",
  "tgt": "( go :ARG1-of ( want :ARG0 ( boy ) ) :ARG0 boy )"
  },
 "randomize_mask-all-mass-unshuffle": {
  "src": "denoise Graph: ( go :ARG1-of <extra_id_0> want :ARG0 ( boy ) ) :ARG0 boy )",
  "tgt": "( want :ARG0 ( boy ) :ARG1 ( go :ARG0 boy ) )"
 },
 "randomize_mask-components": {
  "src": "denoise Graph: ( go :ARG1-of <extra_id_0> want :ARG0 ( boy ) ) :ARG0 boy )",
  "tgt": "<extra_id_0> ( <extra_id_1>"
 },
 "randomize_mask-components-corrupt": {
  "src": "denoise Graph: ( go :ARG1-of ( want :ARG0 ( boy ) ) :ARG0 boy )",
  "tgt": "( go :ARG1-of ( want :ARG0 ( boy ) ) :ARG0 boy )"
 },
 "randomize_mask-components-corrupt-unshuffle": {
  "src": "denoise Graph: ( go :ARG1-of ( want :ARG0 ( boy ) ) :ARG0 boy )",
  "tgt": "( want :ARG0 ( boy ) :ARG1 ( go :ARG0 boy ) )"
 },
 "randomize_mask-components-drop": {
  "src": "denoise Graph: ( go :ARG1-of  want :ARG0 ( boy ) ) :ARG0 boy )",
  "tgt": " ( "
 },
 "randomize_mask-components-mass": {
  "src": "denoise Graph: ( go :ARG1-of <extra_id_0> want :ARG0 ( boy ) ) :ARG0 boy )",
  "tgt": "( go :ARG1-of ( want :ARG0 ( boy ) ) :ARG0 boy )"
 },
 "randomize_mask-components-mass-unshuffle": {
  "src": "denoise Graph: ( go :ARG1-of <extra_id_0> want :ARG0 ( boy ) ) :ARG0 boy )",
  "tgt": "( want :ARG0 ( boy ) :ARG1 ( go :ARG0 boy ) )"
 },
 "randomize_mask-nodes": {
  "src": "denoise Graph: ( <extra_id_0> :ARG1-of ( want :ARG0 ( boy ) ) :ARG0 <extra_id_1> )",
  "tgt": "<extra_id_0> go <extra_id_1> boy <extra_id_2>"
 },
 "randomize_mask-nodes-drop": {
  "src": "denoise Graph: (  :ARG1-of ( want :ARG0 ( boy ) ) :ARG0  )",
  "tgt": " go  boy "
 },
 "randomize_mask-nodes-mass": {
  "src": "denoise Graph: ( <extra_id_0> :ARG1-of ( want :ARG0 ( boy ) ) :ARG0 <extra_id_0> )",
  "tgt": "( go :ARG1-of ( want :ARG0 ( boy ) ) :ARG0 boy )"
  },
 "randomize_mask-nodes-mass-unshuffle": {
  "src": "denoise Graph: ( <extra_id_0> :ARG1-of ( want :ARG0 ( boy ) ) :ARG0 <extra_id_0> )",
  "tgt": "( want :ARG0 ( boy ) :ARG1 ( go :ARG0 boy ) )"
 },
 "randomize_parse-from-triples": {
  "src": "order Graph: <t> want :ARG1 go <t> want :ARG0 boy <t> go :ARG0 boy",
  "tgt": "( go :ARG1-of ( want :ARG0 ( boy ) ) :ARG0 boy )"
 },
 "randomize_reorder": {
  "src": "order Graph: ( go :ARG1-of ( want :ARG0 ( boy ) ) :ARG0 boy )",
  "tgt": "( want :ARG0 ( boy ) :ARG1 ( go :ARG0 boy ) )"
 },
 "reconfigure": {
  "src": "( want :ARG1 ( go :ARG0 boy ) :ARG0 ( boy ) )",
  "tgt": "The boy wants to go."
 },
 "reconfigure_convert-to-triples": {
  "src": "order Graph: ( want :ARG1 ( go :ARG0 boy ) :ARG0 ( boy ) )",
  "tgt": "<t> want :ARG1 go <t> go :ARG0 boy <t> want :ARG0 boy"
 },
 "reconfigure_generate-from-triples": {
  "src": "order Graph: <t> want :ARG1 go <t> go :ARG0 boy <t> want :ARG0 boy",
  "tgt": "The boy wants to go."
 },
 "reconfigure_mask-all": {
  "src": "denoise Graph: ( want :ARG1 <extra_id_0> go :ARG0 boy ) :ARG0 ( boy ) )",
  "tgt": "<extra_id_0> ( <extra_id_1>"
 },
 "reconfigure_mask-all-drop": {
  "src": "denoise Graph: ( want :ARG1  go :ARG0 boy ) :ARG0 ( boy ) )",
  "tgt": " ( "
 },
 "reconfigure_mask-all-mass": {
  "src": "denoise Graph: ( want :ARG1 <extra_id_0> go :ARG0 boy ) :ARG0 ( boy ) )",
  "tgt": "( want :ARG1 ( go :ARG0 boy ) :ARG0 ( boy ) )"
 },
 "reconfigure_mask-all-mass-unshuffle": {
  "src": "denoise Graph: ( want :ARG1 <extra_id_0> go :ARG0 boy ) :ARG0 ( boy ) )",
  "tgt": "( want :ARG0 ( boy ) :ARG1 ( go :ARG0 boy ) )"
 },
 "reconfigure_mask-components": {
  "src": "denoise Graph: ( want :ARG1 <extra_id_0> go :ARG0 boy ) :ARG0 ( boy ) )",
  "tgt": "<extra_id_0> ( <extra_id_1>"
 },
 "reconfigure_mask-components-corrupt": {
  "src": "denoise Graph: ( want :ARG1 ( go :ARG0 boy ) :ARG0 ( boy ) )",
  "tgt": "( want :ARG1 ( go :ARG0 boy ) :ARG0 ( boy ) )"
 },
 "reconfigure_mask-components-corrupt-unshuffle": {
  "src": "denoise Graph: ( want :ARG1 ( go :ARG0 boy ) :ARG0 ( boy ) )",
  "tgt": "( want :ARG0 ( boy ) :ARG1 ( go :ARG0 boy ) )"
 },
 "reconfigure_mask-components-drop": {
  "src": "denoise Graph: ( want :ARG1  go :ARG0 boy ) :ARG0 ( boy ) )",
  "tgt": " ( "
 },
 "reconfigure_mask-components-mass": {
  "src": "denoise Graph: ( want :ARG1 <extra_id_0> go :ARG0 boy ) :ARG0 ( boy ) )",
  "tgt": "( want :ARG1 ( go :ARG0 boy ) :ARG0 ( boy ) )"
 },
 "reconfigure_mask-components-mass-unshuffle": {
  "src": "denoise Graph: ( want :ARG1 <extra_id_0> go :ARG0 boy ) :ARG0 ( boy ) )",
  "tgt": "( want :ARG0 ( boy ) :ARG1 ( go :ARG0 boy ) )"
 },
 "reconfigure_mask-nodes": {
  "src": "denoise Graph: ( <extra_id_0> :ARG1 ( go :ARG0 boy ) :ARG0 ( <extra_id_1> ) )",
  "tgt": "<extra_id_0> want <extra_id_1> boy <extra_id_2>"
 },
 "reconfigure_mask-nodes-drop": {
  "src": "denoise Graph: (  :ARG1 ( go :ARG0 boy ) :ARG0 (  ) )",
  "tgt": " want  boy "
 },
 "reconfigure_mask-nodes-mass": {
  "src": "denoise Graph: ( <extra_id_0> :ARG1 ( go :ARG0 boy ) :ARG0 ( <extra_id_0> ) )",
  "tgt": "( want :ARG1 ( go :ARG0 boy ) :ARG0 ( boy ) )"
 },
 "reconfigure_mask-nodes-mass-unshuffle": {
  "src": "denoise Graph: ( <extra_id_0> :ARG1 ( go :ARG0 boy ) :ARG0 ( <extra_id_0> ) )",
  "tgt": "( want :ARG0 ( boy ) :ARG1 ( go :ARG0 boy ) )"
 },
  "reconfigure_parse-from-triples": {
  "src": "order Graph: <t> want :ARG1 go <t> go :ARG0 boy <t> want :ARG0 boy",
  "tgt": "( want :ARG1 ( go :ARG0 boy ) :ARG0 ( boy ) )"
 },
 "reconfigure_reorder": {
  "src": "order Graph: ( want :ARG1 ( go :ARG0 boy ) :ARG0 ( boy ) )",
  "tgt": "( want :ARG0 ( boy ) :ARG1 ( go :ARG0 boy ) )"
 }
}