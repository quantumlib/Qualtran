import Diagram from 'diagram-js';

import ConnectModule from 'diagram-js/lib/features/connect';
import ContextPadModule from 'diagram-js/lib/features/context-pad';
import CreateModule from 'diagram-js/lib/features/create';
import LassoToolModule from 'diagram-js/lib/features/lasso-tool';
import ModelingModule from 'diagram-js/lib/features/modeling';
import MoveCanvasModule from 'diagram-js/lib/navigation/movecanvas';
import MoveModule from 'diagram-js/lib/features/move';
import OutlineModule from 'diagram-js/lib/features/outline';
import PaletteModule from 'diagram-js/lib/features/palette';
import ResizeModule from 'diagram-js/lib/features/resize';
import RulesModule from 'diagram-js/lib/features/rules';
import SelectionModule from 'diagram-js/lib/features/selection';
import ZoomScrollModule from 'diagram-js/lib/navigation/zoomscroll';


/**
 * A module that changes the default diagram look.
 */
const ElementStyleModule = {
  __init__: [
    [ 'defaultRenderer', function(defaultRenderer) {
      // override default styles
      defaultRenderer.CONNECTION_STYLE = { fill: 'none', strokeWidth: 5, stroke: '#000' };
      defaultRenderer.SHAPE_STYLE = { fill: 'white', stroke: '#000', strokeWidth: 2 };
      defaultRenderer.FRAME_STYLE = { fill: 'none', stroke: '#000', strokeDasharray: 4, strokeWidth: 2 };
    } ]
  ]
};


/**
 * Our editor constructor
 *
 * @param { { container: Element, additionalModules?: Array<any> } } options
 *
 * @return {Diagram}
 */
export default function Editor(options) {

  const {
    container,
    additionalModules = []
  } = options;

  // default modules provided by the toolbox
  const builtinModules = [
    ConnectModule,
    ContextPadModule,
    CreateModule,
    LassoToolModule,
    ModelingModule,
    MoveCanvasModule,
    MoveModule,
    OutlineModule,
    PaletteModule,
    ResizeModule,
    RulesModule,
    SelectionModule,
    ZoomScrollModule
  ];

  // our own modules, contributing controls, customizations, and more
  const customModules = [
    ElementStyleModule
  ];

  return new Diagram({
    canvas: {
      container
    },
    modules: [
      ...builtinModules,
      ...customModules,
      ...additionalModules
    ]
  });
}

// (1) create new editor instance

const diagram = new Editor({
    container: document.querySelector('#container')
  });
  
  
  // (2) draw diagram elements (i.e. import)
  
  const canvas = diagram.get('canvas');
  const elementFactory = diagram.get('elementFactory');
  
  // add root
  var root = elementFactory.createRoot();
  
  canvas.setRootElement(root);
  
  // add shapes
  var shape1 = elementFactory.createShape({
    x: 150,
    y: 100,
    width: 100,
    height: 80
  });
  
  canvas.addShape(shape1, root);
  
  var shape2 = elementFactory.createShape({
    x: 290,
    y: 220,
    width: 100,
    height: 80
  });
  
  canvas.addShape(shape2, root);
  
  
  var connection1 = elementFactory.createConnection({
    waypoints: [
      { x: 250, y: 180 },
      { x: 290, y: 220 }
    ],
    source: shape1,
    target: shape2
  });
  
  canvas.addConnection(connection1, root);
  
  
  var shape3 = elementFactory.createShape({
    x: 450,
    y: 80,
    width: 100,
    height: 80
  });
  
  canvas.addShape(shape3, root);
  
  var shape4 = elementFactory.createShape({
    x: 425,
    y: 50,
    width: 300,
    height: 200,
    isFrame: true
  });
  
  canvas.addShape(shape4, root);
  
  
// (3) interact with the diagram via API
//   const selection = diagram.get('selection');
//   selection.select(shape3);