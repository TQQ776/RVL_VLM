from .mcp_shared import *


class ContainerPlaceMixin:
    def _tool_place_relative_to_object(self, arguments: Dict) -> Tuple[bool, str, Dict]:
        held_object_name = self._normalize_object_name(
            str(
                arguments.get('held_object_name')
                or arguments.get('object_name')
                or arguments.get('held')
                or ''
            )
        )
        return_home_after_place = self._as_bool(arguments.get('return_home_after_place', True))
        reference_name = self._normalize_object_name(
            str(
                arguments.get('reference_object_name')
                or arguments.get('reference')
                or arguments.get('target_object_name')
                or arguments.get('target')
                or ''
            )
        )
        direction = self._normalize_object_name(str(arguments.get('direction') or ''))
        distance_cm = self._place_distance_cm(arguments)
        if not reference_name:
            return False, 'place_relative_to_object failed: reference object name is empty', {}
        if not direction:
            return False, 'place_relative_to_object failed: direction is empty', {}
        reference = self._lookup_scene_object(reference_name)
        if reference is None:
            return False, (
                f'place_relative_to_object failed: no scene memory for "{reference_name}". '
                'Run observe_scene/list_api_objects/box_api_object while the reference is visible first.'
            ), {'scene_memory': self._scene_memory_snapshot()}

        offset = self._direction_offset_m(direction, distance_cm)
        if offset is None:
            return False, (
                f'place_relative_to_object failed: unsupported direction "{direction}". '
                'Use left/right/front/back/up/down or 左/右/前/后/上/下.'
            ), {}

        if not self.motion_lock.acquire(blocking=False):
            return False, 'place_relative_to_object failed: motion already running', {}
        gripper_acquired = False
        try:
            if not self.gripper_lock.acquire(blocking=False):
                return False, 'place_relative_to_object failed: gripper action already running', {}
            gripper_acquired = True
            success, message, result = self._place_relative_to_scene_object(
                held_object_name,
                reference,
                direction,
                distance_cm,
                offset,
            )
            if success and return_home_after_place:
                home_ok, home_message, home_result = self._go_home_half_speed()
                result = dict(result or {})
                result['return_home'] = home_result
                if not home_ok:
                    return False, (
                        f'place_relative_to_object placed object but failed returning home: '
                        f'{home_message}'
                    ), result
                return True, f'{message}; then returned home.', result
            return success, message, result
        finally:
            if gripper_acquired:
                self.gripper_lock.release()
            self.motion_lock.release()

    def _tool_pick_and_place_relative(self, arguments: Dict) -> Tuple[bool, str, Dict]:
        object_name = self._normalize_object_name(
            str(arguments.get('object_name') or arguments.get('held_object_name') or '')
        )
        reference_name = self._normalize_object_name(
            str(arguments.get('reference_object_name') or arguments.get('reference') or '')
        )
        direction = self._normalize_object_name(str(arguments.get('direction') or ''))
        distance_cm = self._place_distance_cm(arguments)
        if not object_name:
            return False, 'pick_and_place_relative failed: object_name is empty', {}
        if not reference_name:
            return False, 'pick_and_place_relative failed: reference_object_name is empty', {}
        if not direction:
            return False, 'pick_and_place_relative failed: direction is empty', {}

        held_entry = self._current_held_object_entry()
        if held_entry is not None:
            if not self._held_entry_matches_name(held_entry, object_name):
                held_label = self._held_entry_label(held_entry)
                return False, (
                    f'pick_and_place_relative failed: gripper already holds "{held_label}", '
                    f'not requested "{object_name}". Use place_relative_to_object if this is intended.'
                ), {'held_object': held_entry}
            place_ok, place_message, place_result = self._tool_place_relative_to_object({
                'held_object_name': self._held_entry_label(held_entry) or object_name,
                'reference_object_name': reference_name,
                'direction': direction,
                'distance_cm': distance_cm,
                'return_home_after_place': False,
            })
            if not place_ok:
                return False, (
                    f'pick_and_place_relative failed while already holding object: {place_message}'
                ), {'already_holding': True, 'held_object': held_entry, 'place': place_result}
            home_ok, home_message, home_result = self._go_home_half_speed()
            if not home_ok:
                return False, (
                    f'pick_and_place_relative placed object while already holding it but failed '
                    f'returning home: {home_message}'
                ), {
                    'already_holding': True,
                    'held_object': held_entry,
                    'place': place_result,
                    'return_home': home_result,
                }
            return True, (
                f'pick_and_place_relative success: already holding "{object_name}", skipped '
                f'observe/grasp and placed it {direction} {distance_cm:.1f}cm relative to '
                f'"{reference_name}", then returned home.'
            ), {
                'already_holding': True,
                'held_object': held_entry,
                'place': place_result,
                'return_home': home_result,
            }

        observe_ok, observe_message, observe_result = self._tool_observe_scene({
            'object_names': [object_name, reference_name],
            'question': (
                f'请优先框出“{object_name}”和“{reference_name}”。'
                '同时请把画面中其他清晰可见、可能成为机械臂运动障碍物的独立物体也分别返回。'
            ),
        })
        if not observe_ok:
            return False, f'pick_and_place_relative failed before grasp: {observe_message}', {
                'observe_scene': observe_result,
            }

        grab_args = {'object_name': object_name, 'return_home_after_grasp': False}
        speed = self._optional_motion_speed(arguments)
        if speed is not None:
            grab_args['motion_speed'] = speed
        grab_ok, grab_message, grab_result = self._tool_grab_api_object(grab_args)
        if not grab_ok:
            return False, f'pick_and_place_relative failed during grasp: {grab_message}', {
                'observe_scene': observe_result,
                'grab': grab_result,
            }

        place_ok, place_message, place_result = self._tool_place_relative_to_object({
            'held_object_name': object_name,
            'reference_object_name': reference_name,
            'direction': direction,
            'distance_cm': distance_cm,
            'return_home_after_place': False,
        })
        if not place_ok:
            return False, f'pick_and_place_relative failed during place: {place_message}', {
                'observe_scene': observe_result,
                'grab': grab_result,
                'place': place_result,
            }
        home_ok, home_message, home_result = self._go_home_half_speed()
        if not home_ok:
            return False, f'pick_and_place_relative placed object but failed returning home: {home_message}', {
                'observe_scene': observe_result,
                'grab': grab_result,
                'place': place_result,
                'return_home': home_result,
            }
        return True, (
            f'pick_and_place_relative success: grabbed "{object_name}" and placed it '
            f'{direction} {distance_cm:.1f}cm relative to "{reference_name}", then returned home.'
        ), {
            'observe_scene': observe_result,
            'grab': grab_result,
            'place': place_result,
            'return_home': home_result,
        }

    def _tool_place_into_container(self, arguments: Dict) -> Tuple[bool, str, Dict]:
        held_object_name = self._normalize_object_name(
            str(
                arguments.get('held_object_name')
                or arguments.get('object_name')
                or arguments.get('held')
                or ''
            )
        )
        return_home_after_place = self._as_bool(arguments.get('return_home_after_place', True))
        container_name = self._normalize_object_name(
            str(
                arguments.get('container_name')
                or arguments.get('box_name')
                or arguments.get('reference_object_name')
                or arguments.get('target')
                or 'box'
            )
        )
        if not container_name:
            return False, 'place_into_container failed: container_name is empty', {}
        container = self._lookup_container_scene_object(container_name)
        if container is None:
            return False, (
                f'place_into_container failed: no scene memory for "{container_name}". '
                'Run observe_scene/list_api_objects/box_api_object while the container is visible first.'
            ), {'scene_memory': self._scene_memory_snapshot()}

        if not self.motion_lock.acquire(blocking=False):
            return False, 'place_into_container failed: motion already running', {}
        gripper_acquired = False
        try:
            if not self.gripper_lock.acquire(blocking=False):
                return False, 'place_into_container failed: gripper action already running', {}
            gripper_acquired = True
            success, message, result = self._place_into_container_from_scene(
                held_object_name,
                container,
            )
            if success and return_home_after_place:
                home_ok, home_message, home_result = self._go_home_half_speed()
                result = dict(result or {})
                result['return_home'] = home_result
                if not home_ok:
                    return False, (
                        f'place_into_container placed object but failed returning home: '
                        f'{home_message}'
                    ), result
                return True, f'{message}; then returned home.', result
            return success, message, result
        finally:
            if gripper_acquired:
                self.gripper_lock.release()
            self.motion_lock.release()

    def _tool_pick_and_place_into_container(self, arguments: Dict) -> Tuple[bool, str, Dict]:
        object_name = self._normalize_object_name(
            str(arguments.get('object_name') or arguments.get('held_object_name') or '')
        )
        container_name = self._normalize_object_name(
            str(arguments.get('container_name') or arguments.get('box_name') or 'box')
        )
        if not object_name:
            return False, 'pick_and_place_into_container failed: object_name is empty', {}
        if not container_name:
            return False, 'pick_and_place_into_container failed: container_name is empty', {}

        held_entry = self._current_held_object_entry()
        if held_entry is not None:
            if not self._held_entry_matches_name(held_entry, object_name):
                held_label = self._held_entry_label(held_entry)
                return False, (
                    f'pick_and_place_into_container failed: gripper already holds "{held_label}", '
                    f'not requested "{object_name}". Use place_into_container if this is intended.'
                ), {'held_object': held_entry}
            place_ok, place_message, place_result = self._tool_place_into_container({
                'held_object_name': self._held_entry_label(held_entry) or object_name,
                'container_name': container_name,
                'return_home_after_place': False,
            })
            if not place_ok:
                return False, (
                    f'pick_and_place_into_container failed while already holding object: '
                    f'{place_message}'
                ), {'already_holding': True, 'held_object': held_entry, 'place': place_result}
            home_ok, home_message, home_result = self._go_home_half_speed()
            if not home_ok:
                return False, (
                    f'pick_and_place_into_container placed object while already holding it but '
                    f'failed returning home: {home_message}'
                ), {
                    'already_holding': True,
                    'held_object': held_entry,
                    'place': place_result,
                    'return_home': home_result,
                }
            return True, (
                f'pick_and_place_into_container success: already holding "{object_name}", '
                f'skipped observe/grasp and placed it into "{container_name}", then returned home.'
            ), {
                'already_holding': True,
                'held_object': held_entry,
                'place': place_result,
                'return_home': home_result,
            }

        observe_ok, observe_message, observe_result = self._tool_observe_scene({
            'object_names': [object_name, container_name],
            'question': (
                f'请优先框出“{object_name}”和用于放置的开口箱子“{container_name}”。'
                '箱子请框外轮廓，不要只框内部。'
                '同时请把画面中其他清晰可见、可能成为机械臂运动障碍物的独立物体也分别返回。'
            ),
        })
        if not observe_ok:
            return False, f'pick_and_place_into_container failed before grasp: {observe_message}', {
                'observe_scene': observe_result,
            }

        grab_args = {'object_name': object_name, 'return_home_after_grasp': False}
        speed = self._optional_motion_speed(arguments)
        if speed is not None:
            grab_args['motion_speed'] = speed
        grab_ok, grab_message, grab_result = self._tool_grab_api_object(grab_args)
        if not grab_ok:
            return False, f'pick_and_place_into_container failed during grasp: {grab_message}', {
                'observe_scene': observe_result,
                'grab': grab_result,
            }

        place_ok, place_message, place_result = self._tool_place_into_container({
            'held_object_name': object_name,
            'container_name': container_name,
            'return_home_after_place': False,
        })
        if not place_ok:
            return False, f'pick_and_place_into_container failed during place: {place_message}', {
                'observe_scene': observe_result,
                'grab': grab_result,
                'place': place_result,
            }
        home_ok, home_message, home_result = self._go_home_half_speed()
        if not home_ok:
            return False, (
                f'pick_and_place_into_container placed object but failed returning home: {home_message}'
            ), {
                'observe_scene': observe_result,
                'grab': grab_result,
                'place': place_result,
                'return_home': home_result,
            }
        return True, (
            f'pick_and_place_into_container success: grabbed "{object_name}" and placed it '
            f'into "{container_name}", then returned home.'
        ), {
            'observe_scene': observe_result,
            'grab': grab_result,
            'place': place_result,
            'return_home': home_result,
        }

    def _tool_pick_all_fruits_into_container(self, arguments: Dict) -> Tuple[bool, str, Dict]:
        container_name = self._normalize_object_name(
            str(arguments.get('container_name') or arguments.get('box_name') or 'box')
        )
        task_request = self._normalize_object_name(
            str(
                arguments.get('task_request')
                or arguments.get('user_request')
                or arguments.get('request')
                or ''
            )
        )
        if not task_request:
            task_request = f'把所有水果放进{container_name}'
        task_request = self._normalize_collection_task_request(task_request)
        max_items = max(1, int(arguments.get('max_items') or 10))
        completed = []
        observations = []
        verifications = []
        semantic_plans = []
        completed_text = lambda: self._completed_object_names_text(completed)
        for index in range(max_items):
            observe_ok, observe_message, observe_result = self._tool_observe_scene({
                'question': (
                    f'原始机器人任务是：“{task_request}”。目标容器是“{container_name}”。'
                    '请框出当前画面中所有清晰可见的独立物体，包括可能符合任务的目标、'
                    '目标容器本身，以及其他可能成为机械臂运动障碍物的物体。'
                    '不要在检测阶段提前过滤；容器内外的物体都可以返回，后续会再根据任务语义判断。'
                    '每个独立物体都必须单独给框，容器请框外轮廓。'
                ),
            })
            observations.append({
                'success': observe_ok,
                'message': observe_message,
                'result': observe_result,
            })
            if not observe_ok:
                verify_ok, verify_message, verify_result = self._verify_all_fruits_in_container(
                    container_name,
                    task_request,
                )
                verifications.append({
                    'success': verify_ok,
                    'message': verify_message,
                    'result': verify_result,
                    'after': 'observe_failed',
                })
                if verify_ok:
                    return True, (
                        f'pick_all_fruits_into_container success: vision confirmed all visible '
                        f'fruits are in "{container_name}" after placing {len(completed)} '
                        f'fruit(s){completed_text()}.'
                    ), {
                        'completed': completed,
                        'observations': observations,
                        'verifications': verifications,
                        'semantic_plans': semantic_plans,
                    }
                prefix = (
                    f'pick_all_fruits_into_container partial failure: placed {len(completed)} '
                    f'fruit(s), but could not confirm the remaining scene'
                    if completed
                    else 'pick_all_fruits_into_container failed before loop'
                )
                return False, f'{prefix}: {observe_message}', {
                    'completed': completed,
                    'observations': observations,
                    'verifications': verifications,
                    'semantic_plans': semantic_plans,
                }

            current_entries = self._remembered_entries_from_observe_result(observe_result)
            plan_ok, plan_message, semantic_plan = self._plan_next_container_task_target(
                task_request,
                container_name,
                current_entries,
            )
            semantic_plans.append({
                'success': plan_ok,
                'message': plan_message,
                'result': semantic_plan,
                'iteration': index,
            })
            if not plan_ok:
                verify_ok, verify_message, verify_result = self._verify_all_fruits_in_container(
                    container_name,
                    task_request,
                )
                verifications.append({
                    'success': verify_ok,
                    'message': verify_message,
                    'result': verify_result,
                    'after': 'semantic_plan_failed',
                })
                if verify_ok:
                    return True, (
                        f'pick_all_fruits_into_container success: vision confirmed all visible '
                        f'fruits are in "{container_name}"; placed {len(completed)} '
                        f'fruit(s){completed_text()}.'
                    ), {
                        'completed': completed,
                        'observations': observations,
                        'verifications': verifications,
                        'semantic_plans': semantic_plans,
                    }
                return False, (
                    f'pick_all_fruits_into_container could not plan the next task target: '
                    f'{plan_message}; final verification: {verify_message}'
                ), {
                    'completed': completed,
                    'observations': observations,
                    'verifications': verifications,
                    'semantic_plans': semantic_plans,
                }

            if self._semantic_plan_task_complete(semantic_plan):
                return True, (
                    f'pick_all_fruits_into_container success: vision confirmed the original task '
                    f'is complete after placing {len(completed)} object(s){completed_text()}.'
                ), {
                    'completed': completed,
                    'observations': observations,
                    'verifications': verifications,
                    'semantic_plans': semantic_plans,
                }

            fruit = self._target_entry_from_semantic_plan(
                semantic_plan,
                container_name,
                current_entries,
                task_request,
            )
            if fruit is None:
                verify_ok, verify_message, verify_result = self._verify_all_fruits_in_container(
                    container_name,
                    task_request,
                )
                verifications.append({
                    'success': verify_ok,
                    'message': verify_message,
                    'result': verify_result,
                    'after': 'semantic_plan_no_matched_detection',
                })
                if verify_ok:
                    return True, (
                        f'pick_all_fruits_into_container success: vision confirmed all task targets '
                        f'are already in "{container_name}"; placed {len(completed)} '
                        f'object(s){completed_text()}.'
                    ), {
                        'completed': completed,
                        'observations': observations,
                        'verifications': verifications,
                        'semantic_plans': semantic_plans,
                    }
                return False, (
                    f'pick_all_fruits_into_container semantic planner did not provide a matched '
                    f'outside target detection; final verification: {verify_message}'
                ), {
                    'completed': completed,
                    'observations': observations,
                    'verifications': verifications,
                    'semantic_plans': semantic_plans,
                }

            object_name = self._held_entry_label(fruit)
            if not object_name:
                object_name = str(fruit.get('class_name') or fruit.get('label_zh') or f'target_{index + 1}')
            pick_ok, pick_message, pick_result = self._tool_pick_and_place_into_container({
                'object_name': object_name,
                'container_name': container_name,
                'motion_speed': arguments.get('motion_speed', None),
            })
            completed.append({
                'object_name': object_name,
                'success': pick_ok,
                'message': pick_message,
                'result': pick_result,
            })
            if not pick_ok:
                held_entry = self._current_held_object_entry()
                if held_entry is not None and self._held_entry_matches_name(held_entry, object_name):
                    place_ok, place_message, place_result = self._tool_place_into_container({
                        'held_object_name': self._held_entry_label(held_entry) or object_name,
                        'container_name': container_name,
                        'return_home_after_place': False,
                    })
                    completed[-1]['recovery_place'] = {
                        'success': place_ok,
                        'message': place_message,
                        'result': place_result,
                    }
                    if place_ok:
                        home_ok, home_message, home_result = self._go_home_half_speed()
                        completed[-1]['recovery_return_home'] = {
                            'success': home_ok,
                            'message': home_message,
                            'result': home_result,
                        }
                        if not home_ok:
                            return False, (
                                f'pick_all_fruits_into_container placed "{object_name}" after '
                                f'a partial failure but failed returning home: {home_message}'
                            ), {
                                'completed': completed,
                                'observations': observations,
                                'verifications': verifications,
                                'semantic_plans': semantic_plans,
                            }
                        continue
                    return False, (
                        f'pick_all_fruits_into_container grasped "{object_name}" but failed placing '
                        f'it into "{container_name}": {place_message}'
                    ), {
                        'completed': completed,
                        'observations': observations,
                        'verifications': verifications,
                        'semantic_plans': semantic_plans,
                    }
                verify_ok, verify_message, verify_result = self._verify_all_fruits_in_container(
                    container_name,
                    task_request,
                )
                verifications.append({
                    'success': verify_ok,
                    'message': verify_message,
                    'result': verify_result,
                    'after': f'pick_failed:{object_name}',
                })
                if verify_ok:
                    return True, (
                        f'pick_all_fruits_into_container success: "{object_name}" action reported '
                        f'an error, but vision confirmed all visible fruits are already in '
                        f'"{container_name}" after placing {len(completed)} '
                        f'object(s){completed_text()}.'
                    ), {
                        'completed': completed,
                        'observations': observations,
                        'verifications': verifications,
                        'semantic_plans': semantic_plans,
                    }
                return False, (
                    f'pick_all_fruits_into_container failed on "{object_name}": {pick_message}'
                ), {
                    'completed': completed,
                    'observations': observations,
                    'verifications': verifications,
                    'semantic_plans': semantic_plans,
                }

        verify_ok, verify_message, verify_result = self._verify_all_fruits_in_container(
            container_name,
            task_request,
        )
        verifications.append({
            'success': verify_ok,
            'message': verify_message,
            'result': verify_result,
            'after': 'max_items',
        })
        if verify_ok:
            return True, (
                f'pick_all_fruits_into_container success: reached max_items={max_items}, but vision '
                f'confirmed all visible fruits are in "{container_name}" after placing '
                f'{len(completed)} object(s){completed_text()}.'
            ), {
                'completed': completed,
                'observations': observations,
                'verifications': verifications,
                'semantic_plans': semantic_plans,
            }
        return False, (
            f'pick_all_fruits_into_container stopped after max_items={max_items}; '
            f'placed {len(completed)} fruit(s), but did not confirm all outside fruits are done.'
        ), {
            'completed': completed,
            'observations': observations,
            'verifications': verifications,
            'semantic_plans': semantic_plans,
        }

    @staticmethod
    def _completed_object_names_text(completed: List[Dict]) -> str:
        names = []
        for item in completed:
            if not item.get('success'):
                continue
            name = str(item.get('object_name') or '').strip()
            if name and name not in names:
                names.append(name)
        if not names:
            return ''
        return ': ' + ', '.join(names)

    def _plan_next_container_task_target(
        self,
        task_request: str,
        container_name: str,
        entries: List[Dict],
    ) -> Tuple[bool, str, Dict]:
        if not self.vision_enabled:
            return False, 'semantic task planning failed: vision is disabled', {}
        entry_summary = self._scene_entries_for_llm(entries)
        prompt = (
            '你是机器人长序列任务规划器。请根据原始用户任务、当前相机画面、'
            '以及当前检测框列表，判断下一步应该搬运哪个目标。\n'
            f'原始用户任务：{task_request}\n'
            f'目标容器名称：{container_name}\n'
            f'当前检测框列表 JSON：{json.dumps(entry_summary, ensure_ascii=False)}\n'
            '请完全按原始用户任务的语义判断哪些可见物体属于本次任务目标；'
            '例如用户说“所有水果”，就由你根据常识和画面判断哪些可见物体是日常食用水果。'
            '这里的“水果/all fruits”按日常饮食分类理解，例如苹果、橘子、香蕉、梨、葡萄等；'
            '不要按植物学果实概念扩大范围。辣椒、青椒、甜椒、番茄、黄瓜、茄子等'
            '日常作为蔬菜或调味食材的物体，必须放进 ignored_objects，不能作为 next_target。'
            '非本次任务目标即使在容器外，也必须放进 ignored_objects，不能因为它们在外面判失败。'
            '如果还有本次任务目标在容器外，请选择一个最清晰、最适合当前抓取的目标作为 next_target。'
            'next_target_id 必须优先使用当前检测框列表里的 id；next_target_index 使用列表里的 0 基 index。'
            '如果所有本次任务目标都已经在容器内部或开口范围内，task_complete 必须为 true，'
            'next_target 置为空字符串。'
            '只输出严格 JSON，不要 Markdown，不要解释。'
            'JSON 格式：'
            '{"task_complete":true或false,'
            '"container_visible":true或false,'
            '"next_target":"目标名或空字符串",'
            '"next_target_id":"检测框id或空字符串",'
            '"next_target_index":整数或-1,'
            '"outside_targets":["仍在容器外且属于任务目标的物体名"],'
            '"inside_targets":["已经在容器内且属于任务目标的物体名"],'
            '"ignored_objects":["可见但不属于本次任务目标的物体名"],'
            '"reason":"一句中文原因"}'
        )
        try:
            data = self._ask_vision_json(prompt)
        except Exception as exc:
            return False, f'semantic task planning failed: {exc}', {}
        data['_detected_entries'] = entry_summary
        if self._semantic_plan_task_complete(data):
            return True, 'semantic task planning success: task already complete', data
        if not self._semantic_plan_candidate_names(data) and not self._semantic_plan_candidate_ids(data):
            return False, 'semantic task planning failed: no outside task target was selected', data
        return True, 'semantic task planning success: selected next target', data

    def _semantic_plan_task_complete(self, data: Dict) -> bool:
        if not isinstance(data, dict):
            return False
        complete = self._as_bool(
            data.get('task_complete', data.get('all_fruits_in_container', False))
        )
        outside = data.get('outside_targets', data.get('outside_fruits', []))
        if not isinstance(outside, list):
            outside = [outside] if outside else []
        outside = [str(item).strip() for item in outside if str(item).strip()]
        return bool(complete and not outside)

    def _target_entry_from_semantic_plan(
        self,
        data: Dict,
        container_name: str,
        entries: List[Dict],
        task_request: str = '',
    ) -> Optional[Dict]:
        container = self._lookup_container_scene_object(container_name)
        container_model = self._container_model_from_scene(container) if container is not None else None

        for target_id in self._semantic_plan_candidate_ids(data):
            for entry in entries:
                if str(entry.get('id') or '') != target_id:
                    continue
                if self._semantic_target_entry_allowed(entry, container_model, task_request):
                    return copy.deepcopy(entry)

        target_index = self._semantic_plan_target_index(data)
        if target_index is not None and 0 <= target_index < len(entries):
            entry = entries[target_index]
            if self._semantic_target_entry_allowed(entry, container_model, task_request):
                return copy.deepcopy(entry)

        for name in self._semantic_plan_candidate_names(data):
            for entry in entries:
                if not self._held_entry_matches_name(entry, name):
                    continue
                if self._semantic_target_entry_allowed(entry, container_model, task_request):
                    return copy.deepcopy(entry)
        return None

    def _semantic_target_entry_allowed(
        self,
        entry: Dict,
        container_model: Optional[Dict],
        task_request: str = '',
    ) -> bool:
        if self._is_container_entry(entry):
            return False
        if container_model is not None and self._entry_center_inside_container_xy(entry, container_model):
            return False
        if self._is_daily_fruit_task(task_request) and self._entry_is_excluded_from_daily_fruits(entry):
            label = self._held_entry_label(entry) or str(entry.get('class_name') or '')
            self._publish_status(
                f'semantic target rejected for daily-fruit task: "{label}" is treated as non-fruit'
            )
            return False
        return True

    @staticmethod
    def _normalize_collection_task_request(task_request: str) -> str:
        text = str(task_request or '').strip()
        if not text:
            return text
        compact = ''.join(text.split()).lower()
        if '水果' not in compact and 'fruit' not in compact:
            return text
        hint = (
            '（任务口径：这里的“水果/all fruits”按日常食用水果理解；'
            '辣椒、青椒、甜椒、番茄、黄瓜、茄子等日常作为蔬菜或调味食材的物体不属于本任务目标。）'
        )
        return text if hint in text else f'{text}{hint}'

    @staticmethod
    def _is_daily_fruit_task(task_request: str) -> bool:
        compact = ''.join(str(task_request or '').split()).lower()
        return '水果' in compact or 'fruit' in compact

    def _entry_is_excluded_from_daily_fruits(self, entry: Dict) -> bool:
        names = []
        for key in ('class_name', 'label_zh', 'label_en', 'name'):
            value = str(entry.get(key) or '').strip()
            if value:
                names.append(value)
        for name in entry.get('names') or []:
            value = str(name or '').strip()
            if value:
                names.append(value)
        compact = ''.join(names).lower()
        excluded = (
            '辣椒',
            '青椒',
            '红椒',
            '甜椒',
            '彩椒',
            '尖椒',
            '朝天椒',
            'pepper',
            'chili',
            'chilli',
            'capsicum',
            '番茄',
            '西红柿',
            'tomato',
            '黄瓜',
            'cucumber',
            '茄子',
            'eggplant',
            'aubergine',
        )
        return any(token in compact for token in excluded)

    @staticmethod
    def _semantic_plan_target_index(data: Dict) -> Optional[int]:
        if not isinstance(data, dict):
            return None
        raw = data.get('next_target_index')
        if raw is None or raw == '':
            raw = data.get('target_index')
        try:
            index = int(raw)
        except (TypeError, ValueError):
            return None
        return index if index >= 0 else None

    @staticmethod
    def _semantic_plan_candidate_ids(data: Dict) -> List[str]:
        if not isinstance(data, dict):
            return []
        raw_items = [
            data.get('next_target_id'),
            data.get('target_id'),
        ]
        ids = []
        for raw in raw_items:
            value = str(raw or '').strip()
            if value and value not in ids:
                ids.append(value)
        return ids

    @staticmethod
    def _semantic_plan_candidate_names(data: Dict) -> List[str]:
        if not isinstance(data, dict):
            return []
        raw_items = [
            data.get('next_target'),
            data.get('target'),
            data.get('next_object'),
        ]
        outside_targets = data.get('outside_targets', data.get('outside_fruits', []))
        if isinstance(outside_targets, list):
            raw_items.extend(outside_targets)
        elif outside_targets:
            raw_items.append(outside_targets)
        target_objects = data.get('target_objects', [])
        if isinstance(target_objects, list):
            for item in target_objects:
                if isinstance(item, dict):
                    status = str(item.get('status') or item.get('state') or '').lower()
                    if any(token in status for token in ('inside', 'in_container', 'done', '完成', '容器内')):
                        continue
                    raw_items.extend([
                        item.get('name'),
                        item.get('label'),
                        item.get('class_name'),
                        item.get('label_zh'),
                    ])
                else:
                    raw_items.append(item)
        names = []
        for raw in raw_items:
            value = normalize_object_name(str(raw or ''))
            if value and value.lower() not in {'none', 'null', '无', '空'} and value not in names:
                names.append(value)
        return names

    def _verify_all_fruits_in_container(
        self,
        container_name: str,
        task_request: str = '',
    ) -> Tuple[bool, str, Dict]:
        if not self.vision_enabled:
            return False, 'final vision verification failed: vision is disabled', {}
        held_entry = self._current_held_object_entry()
        if held_entry is not None:
            held_label = self._held_entry_label(held_entry) or 'held object'
            return False, (
                f'final vision verification blocked: gripper is still holding "{held_label}", '
                'so the all-fruits-in-container task is not complete.'
            ), {'held_object': held_entry}
        if not task_request:
            task_request = f'把所有水果放进{container_name}'
        task_request = self._normalize_collection_task_request(task_request)
        prompt = (
            '请检查当前相机画面，判断机器人长序列任务是否已经完成。\n'
            f'原始用户任务：{task_request}\n'
            f'目标容器是“{container_name}”，也可能表现为箱子、篮子、筐或盒子。'
            '请完全按原始任务的语义判断哪些可见物体属于本次任务目标。'
            '如果任务说“水果/all fruits”，这里的“水果”按日常食用水果理解，'
            '例如苹果、橘子、香蕉、梨、葡萄等；不要按植物学果实概念扩大范围。'
            '辣椒、青椒、甜椒、番茄、黄瓜、茄子等日常作为蔬菜/调味食材的物体，'
            '在用户只说“水果”时必须放进 ignored_objects，不属于 outside_targets。'
            '只有本次任务目标还在容器外面时，task_complete 才为 false。'
            '非本次任务目标即使在容器外面，也不要放进 outside_targets，'
            '不要因为它们在外面判定失败。'
            '如果所有本次任务目标都已经在容器内部或开口范围内，task_complete 必须为 true。'
            '如果看不清目标容器，container_visible 必须为 false，task_complete 必须为 false。'
            '只输出严格 JSON，不要 Markdown，不要解释。'
            'JSON 格式：'
            '{"task_complete":true或false,'
            '"all_fruits_in_container":true或false,'
            '"container_visible":true或false,'
            '"inside_targets":["已经在容器内且属于任务目标的物体名"],'
            '"outside_targets":["仍在容器外且属于任务目标的物体名"],'
            '"ignored_objects":["可见但不属于本次任务目标的物体名"],'
            '"reason":"一句中文原因"}'
        )
        try:
            data = self._ask_vision_json(prompt)
        except Exception as exc:
            return False, f'final vision verification failed: {exc}', {}
        outside_targets = data.get('outside_targets', data.get('outside_fruits', []))
        if not isinstance(outside_targets, list):
            outside_targets = [outside_targets] if outside_targets else []
        outside_targets = [str(item).strip() for item in outside_targets if str(item).strip()]
        container_visible = self._as_bool(data.get('container_visible', False))
        all_in = self._as_bool(
            data.get('task_complete', data.get('all_fruits_in_container', False))
        )
        reason = str(data.get('reason') or '').strip()
        if all_in and container_visible and not outside_targets:
            return True, (
                f'final vision verification success: all task targets are in "{container_name}". '
                f'{reason}'
            ), data
        return False, (
            f'final vision verification says task is not complete: '
            f'container_visible={container_visible}, outside_targets={outside_targets}. {reason}'
        ), data

    def _place_into_container_from_scene(
        self,
        held_object_name: str,
        container: Dict,
    ) -> Tuple[bool, str, Dict]:
        container_xyz = container.get('base_xyz', [])
        if not isinstance(container_xyz, list) or len(container_xyz) != 3:
            return False, 'place_into_container failed: container memory has no base_xyz', {
                'container': container,
            }
        current_pose = self._current_ee_pose()
        if current_pose is None:
            return False, 'place_into_container failed: cannot read current end-effector pose', {}

        model = self._container_model_from_scene(container)
        ok, message = self._apply_container_collision_model(model)
        steps = [{'step': 'apply_container_collision', 'success': ok, 'message': message}]
        if not ok:
            return False, f'place_into_container failed adding collision model: {message}', {
                'steps': steps,
                'container': container,
                'container_model': model,
            }

        target_xyz = (
            float(model['center_xyz'][0]),
            float(model['center_xyz'][1]),
            float(model['top_z']) + self.place_target_z_offset_m,
        )
        success, message, result = self._execute_unified_place_motion(
            held_object_name,
            target_xyz,
            action_name='place_into_container',
            pre_place_label='container pre-position',
            steps=steps,
        )
        result = dict(result or {})
        result.update({
            'container': container,
            'container_model': model,
        })
        if not success:
            return False, message, result

        message = (
            f'place_into_container success: placed "{held_object_name or "held object"}" into '
            f'"{"/".join(container.get("names", [])[:2]) or container.get("class_name", "container")}".'
        )
        self._publish_status(message)
        return True, message, result

    def _move_z_by_axis_steps(
        self,
        centimeters: float,
        min_fraction: Optional[float] = None,
        avoid_collisions: Optional[bool] = None,
    ) -> Tuple[bool, str, List[Dict]]:
        remaining = float(centimeters)
        limit = max(0.1, abs(float(self.max_single_axis_move_cm)))
        axis_steps = []
        while abs(remaining) > 1e-3:
            step_cm = max(-limit, min(limit, remaining))
            ok, message = self._move_axis(
                'z',
                step_cm,
                min_fraction=min_fraction,
                avoid_collisions=avoid_collisions,
            )
            axis_steps.append({
                'centimeters': step_cm,
                'success': ok,
                'message': message,
            })
            if not ok:
                return False, message, axis_steps
            remaining -= step_cm
        if not axis_steps:
            return True, 'move_z_cm skipped: requested descend is 0.00 cm', axis_steps
        return True, '；'.join(step['message'] for step in axis_steps), axis_steps

    def _execute_unified_place_motion(
        self,
        held_object_name: str,
        target_xyz: Tuple[float, float, float],
        action_name: str,
        pre_place_label: str,
        steps: Optional[List[Dict]] = None,
    ) -> Tuple[bool, str, Dict]:
        steps = list(steps or [])
        current_pose = self._current_ee_pose()
        if current_pose is None:
            return False, f'{action_name} failed: cannot read current end-effector pose', {
                'steps': steps,
            }

        working_pose = copy.deepcopy(current_pose)
        if self.place_lift_m > 0.0:
            lift_cm = self.place_lift_m * 100.0
            ok, message, lift_axis_steps = self._move_z_by_axis_steps(
                lift_cm,
                avoid_collisions=self.place_lift_avoid_collisions,
            )
            lift_pose = self._current_ee_pose()
            steps.append({
                'step': 'lift_before_place',
                'success': ok,
                'message': message,
                'axis_steps': lift_axis_steps,
                'requested_lift_cm': lift_cm,
                'pose': self._pose_summary(lift_pose) if lift_pose is not None else None,
            })
            if not ok:
                return False, f'{action_name} failed during lift before place: {message}', {
                    'held_object_name': held_object_name,
                    'target_xyz': list(target_xyz),
                    'steps': steps,
                }
            if lift_pose is not None:
                working_pose = lift_pose

        if self.place_transfer_via_home:
            self._publish_status(
                f'{action_name}: moving to configured home before unified pre-place motion'
            )
            ok, message = self._go_home()
            home_pose = self._current_ee_pose()
            steps.append({
                'step': 'transfer_via_home',
                'success': ok,
                'message': message,
                'pose': self._pose_summary(home_pose) if home_pose is not None else None,
            })
            if not ok:
                return False, f'{action_name} failed moving to home transfer: {message}', {
                    'held_object_name': held_object_name,
                    'target_xyz': list(target_xyz),
                    'steps': steps,
                }
            if home_pose is not None:
                working_pose = home_pose

        target_pose = copy.deepcopy(working_pose)
        target_pose.pose.position.x = float(target_xyz[0])
        target_pose.pose.position.y = float(target_xyz[1])
        target_pose.pose.position.z = float(target_xyz[2])
        pre_place_pose = copy.deepcopy(target_pose)
        pre_place_pose.pose.position.z += self.place_pre_height_m
        ok, message = self._move_to_pose_with_move_group(pre_place_pose, pre_place_label)
        steps.append({
            'step': 'move_to_pre_place',
            'success': ok,
            'message': message,
            'pose': self._pose_summary(pre_place_pose),
            'pre_height_m': self.place_pre_height_m,
        })
        if not ok:
            return False, f'{action_name} failed moving to pre-place pose: {message}', {
                'held_object_name': held_object_name,
                'target_pose': self._pose_summary(target_pose),
                'pre_place_pose': self._pose_summary(pre_place_pose),
                'steps': steps,
            }

        descend_cm = self.place_descend_m * 100.0
        release_pose = copy.deepcopy(pre_place_pose)
        if descend_cm > 0.0:
            release_pose.pose.position.z -= descend_cm / 100.0
            ok, message, descend_axis_steps = self._move_z_by_axis_steps(
                -descend_cm,
                min_fraction=self.place_descend_min_fraction,
            )
            current_release_pose = self._current_ee_pose()
            if current_release_pose is not None:
                release_pose = current_release_pose
            steps.append({
                'step': 'single_axis_z_descend',
                'success': ok,
                'message': message,
                'axis_steps': descend_axis_steps,
                'requested_descend_cm': descend_cm,
                'pose': self._pose_summary(release_pose),
            })
            if not ok:
                return False, f'{action_name} failed during single-axis Z descend: {message}', {
                    'held_object_name': held_object_name,
                    'target_pose': self._pose_summary(target_pose),
                    'pre_place_pose': self._pose_summary(pre_place_pose),
                    'release_pose': self._pose_summary(release_pose),
                    'steps': steps,
                }
        else:
            steps.append({
                'step': 'single_axis_z_descend',
                'success': True,
                'message': 'skipped: place_descend_m is 0.000m',
                'requested_descend_cm': 0.0,
                'pose': self._pose_summary(release_pose),
            })

        ok, message = self._open_gripper(detach_attached=False)
        steps.append({'step': 'open_gripper', 'success': ok, 'message': message})
        if not ok:
            return False, f'{action_name} failed opening gripper: {message}', {
                'held_object_name': held_object_name,
                'target_pose': self._pose_summary(target_pose),
                'pre_place_pose': self._pose_summary(pre_place_pose),
                'release_pose': self._pose_summary(release_pose),
                'steps': steps,
            }

        detach_ok, detach_message = self._detach_current_held_object()
        steps.append({'step': 'detach_held_object', 'success': detach_ok, 'message': detach_message})
        if not detach_ok:
            return False, f'{action_name} failed detaching held object model: {detach_message}', {
                'held_object_name': held_object_name,
                'target_pose': self._pose_summary(target_pose),
                'pre_place_pose': self._pose_summary(pre_place_pose),
                'release_pose': self._pose_summary(release_pose),
                'steps': steps,
            }

        steps.append({
            'step': 'post_release_motion',
            'success': True,
            'message': 'skipped unified place release lift; caller may return home immediately',
        })
        return True, f'{action_name} unified place motion completed', {
            'held_object_name': held_object_name,
            'target_pose': self._pose_summary(target_pose),
            'pre_place_pose': self._pose_summary(pre_place_pose),
            'release_pose': self._pose_summary(release_pose),
            'steps': steps,
        }

    def _place_relative_to_scene_object(
        self,
        held_object_name: str,
        reference: Dict,
        direction: str,
        distance_cm: float,
        offset_m: Tuple[float, float, float],
    ) -> Tuple[bool, str, Dict]:
        reference_xyz = reference.get('base_xyz', [])
        if not isinstance(reference_xyz, list) or len(reference_xyz) != 3:
            return False, 'place_relative_to_object failed: reference memory has no base_xyz', {
                'reference': reference,
            }
        target_xyz = (
            float(reference_xyz[0]) + float(offset_m[0]),
            float(reference_xyz[1]) + float(offset_m[1]),
            float(reference_xyz[2]) + self.place_target_z_offset_m + float(offset_m[2])
        )
        success, message, result = self._execute_unified_place_motion(
            held_object_name,
            target_xyz,
            action_name='place_relative_to_object',
            pre_place_label='place pre-position',
        )
        result = dict(result or {})
        result.update({
            'reference_object': reference,
            'direction': direction,
            'distance_cm': distance_cm,
            'offset_m': list(offset_m),
        })
        if not success:
            return False, message, result

        message = (
            f'place_relative_to_object success: placed "{held_object_name or "held object"}" '
            f'{direction} {distance_cm:.1f}cm relative to '
            f'"{"/".join(reference.get("names", [])[:2]) or reference.get("class_name", "object")}".'
        )
        self._publish_status(message)
        return True, message, result

    def _place_distance_cm(self, arguments: Dict) -> float:
        raw = arguments.get('distance_cm')
        if raw is None or raw == '':
            raw = arguments.get('centimeters')
        if raw is None or raw == '':
            return float(self.place_default_offset_cm)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return float(self.place_default_offset_cm)
        return abs(value)

    def _current_held_object_entry(self) -> Optional[Dict]:
        with self.held_object_lock:
            if not self.current_attached_object_id:
                return None
            return copy.deepcopy(self.held_object_visual)

    def _held_entry_label(self, entry: Optional[Dict]) -> str:
        if not isinstance(entry, dict):
            return ''
        names = entry.get('names', [])
        if isinstance(names, list) and names:
            return self._normalize_object_name(str(names[0]))
        return self._normalize_object_name(
            str(entry.get('class_name') or entry.get('label_zh') or '')
        )

    def _held_entry_matches_name(self, entry: Optional[Dict], name: str) -> bool:
        requested_key = self._scene_name_key(name)
        if not requested_key:
            return True
        if not isinstance(entry, dict):
            return False
        names = [
            self._scene_name_key(item)
            for item in entry.get('names', [])
        ]
        names.extend([
            self._scene_name_key(str(entry.get('class_name') or '')),
            self._scene_name_key(str(entry.get('label_zh') or '')),
        ])
        names = [item for item in names if item]
        return any(
            requested_key == item
            or requested_key in item
            or item in requested_key
            for item in names
        )

    @staticmethod
    def _direction_offset_m(direction: str, distance_cm: float) -> Optional[Tuple[float, float, float]]:
        text = ''.join(str(direction or '').strip().lower().split())
        distance_m = abs(float(distance_cm)) / 100.0
        if not text:
            return None
        if text in ('x+', '+x', '右', '右边', '右方', 'right'):
            return (distance_m, 0.0, 0.0)
        if text in ('x-', '-x', '左', '左边', '左方', 'left'):
            return (-distance_m, 0.0, 0.0)
        if text in ('y+', '+y', '前', '前面', '前方', 'forward', 'front'):
            return (0.0, distance_m, 0.0)
        if text in ('y-', '-y', '后', '後', '后面', '後面', '后方', '後方', 'backward', 'back'):
            return (0.0, -distance_m, 0.0)
        if text in ('z+', '+z', '上', '上面', '上方', 'up', 'above', 'top'):
            return (0.0, 0.0, distance_m)
        if text in ('z-', '-z', '下', '下面', '下方', 'down', 'below', 'bottom'):
            return (0.0, 0.0, -distance_m)
        return None

    def place_relative_callback(
        self,
        request: PlaceRelative.Request,
        response: PlaceRelative.Response,
    ):
        arguments = {
            'held_object_name': str(request.held_object_name or ''),
            'reference_object_name': str(request.reference_object_name or ''),
            'direction': str(request.direction or ''),
            'distance_cm': float(request.distance_cm),
        }
        success, message, result = self._tool_place_relative_to_object(arguments)
        response.success = bool(success)
        response.message = message
        response.result_json = json.dumps(result or {}, ensure_ascii=False)
        return response

    def place_into_container_callback(
        self,
        request: PlaceIntoContainer.Request,
        response: PlaceIntoContainer.Response,
    ):
        arguments = {
            'held_object_name': str(request.held_object_name or ''),
            'container_name': str(request.container_name or ''),
        }
        success, message, result = self._tool_place_into_container(arguments)
        response.success = bool(success)
        response.message = message
        response.result_json = json.dumps(result or {}, ensure_ascii=False)
        return response
