#include "src/commands/addnode.cpp"
#include "src/commands/control.cpp"
#include "src/commands/extract.cpp"
#include "src/commands/model.cpp"
#include "src/commands/project.cpp"
#include "src/commands/property.cpp"
#include "src/commands/copy_node.cpp"
#include "src/commands/delnode.cpp"

#include "src/models/model.cpp"
#include "src/models/em_model.cpp"
#include "src/models/lwr_model.cpp"
#include "src/models/null.cpp"
#include "src/models/classifier.cpp"
#include "src/models/dtree.cpp"
#include "src/models/em.cpp"
#include "src/models/foil.cpp"
#include "src/models/lda.cpp"
#include "src/models/linear.cpp"
#include "src/models/lwr.cpp"
#include "src/models/mode.cpp"
#include "src/models/nn.cpp"

#include "src/filters/absval.cpp"
#include "src/filters/bbox.cpp"
#include "src/filters/compare.cpp"
#include "src/filters/direction.cpp"
#include "src/filters/distance.cpp"
#include "src/filters/distance_xyz.cpp"
#include "src/filters/dist_select_xyz.cpp"
#include "src/filters/has_property.cpp"
#include "src/filters/intersect.cpp"
#include "src/filters/node.cpp"
#include "src/filters/occlusion.cpp"
#include "src/filters/ontop.cpp"
#include "src/filters/overlap.cpp"
#include "src/filters/ptlist.cpp"
#include "src/filters/stats.cpp"
#include "src/filters/vec3.cpp"
#include "src/filters/monitor_object.cpp"
#include "src/filters/higher_than.cpp"

#include "src/change_tracking_list.cpp"
#include "src/cliproxy.cpp"
#include "src/command.cpp"
#include "src/command_table.cpp"
#include "src/common.cpp"
#include "src/drawer.cpp"
#include "src/filter.cpp"
#include "src/filter_input.cpp"
#include "src/filter_table.cpp"
#include "src/logger.cpp"
#include "src/mat.cpp"
#include "src/relation.cpp"
#include "src/scene.cpp"
#include "src/scene_sig.cpp"
#include "src/serialize.cpp"
#include "src/sgnode.cpp"
#include "src/soar_interface.cpp"
#include "src/svs.cpp"
#include "src/timer.cpp"
