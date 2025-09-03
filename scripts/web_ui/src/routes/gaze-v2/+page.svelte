<script lang="ts">
	import { Loader } from '@lucide/svelte';
	import DemoPage from '$lib/pages/DemoPage/index.svelte';
	import { demos } from '$lib/demos';
	import { cn } from '$lib/utils';
	import io from 'socket.io-client';
	import { ScrollArea } from '$lib/components/ui/scroll-area/index';
	import * as Card from '$lib/components/ui/card/index';
	import * as Table from '$lib/components/ui/table/index';
	import { onMount } from 'svelte';

	const pinned_buffer_size = 10;

	let loading = $state(true);
	let frame = $state('');
	let metadata: { [key: string]: any } = $state({});
	let pinned_track_ids: number[] = $state([]);
	let detections = $derived.by(() => {
		if (!Object.keys(metadata).length) {
			return [];
		}
		let all_detections: any[] = metadata.current_frame_data.tracks;
		if (!all_detections) {
			return [];
		}
		let new_detections: any[] = [];
		for (const pinned_track_id of pinned_track_ids) {
			//@ts-ignore
			let pinned_detection = all_detections.find(
				(detection) => detection.track_id === pinned_track_id
			);
			if (pinned_detection) {
				new_detections.push(pinned_detection);
				all_detections = all_detections.filter(
					(detection) => detection.track_id !== pinned_track_id
				);
			}
		}
		new_detections = new_detections.concat(
			all_detections.toSorted((a: any, b: any) => {
				return a.track_id - b.track_id;
			})
		);
		return new_detections;
	});
	let historical_people_tracks = $derived.by(() => {
		if (!Object.keys(metadata).length) {
			return [];
		}
		if (!metadata.historical_data.tracks) {
			return [];
		}
		return metadata.historical_data.tracks
			.map(
				(last_detection: {
					track_id: number;
					enter_time: string;
					exit_time: string;
					longest_gaze: number;
					face_img: string;
					gender: string;
					age: number;
					accurate: boolean;
				}) => {
					return {
						...last_detection,
						enter_time: new Date(last_detection.enter_time),
						exit_time: new Date(last_detection.exit_time)
					};
				}
			)
			.sort((a: any, b: any) => {
				return b.exit_time - a.exit_time;
			});
	});

	let fps: number[] = $state([]);
	let fps_count = $derived.by(() => {
		if (fps.length > 2) {
			return ((fps.length - 1) / (fps[fps.length - 1] - fps[0])) * 1000;
		} else {
			return 'Need more frames';
		}
	});

	function formatDateTime(date: Date): string {
		const day = String(date.getDate()).padStart(2, '0');
		const month = String(date.getMonth() + 1).padStart(2, '0'); // Month is 0-indexed
		const year = date.getFullYear();

		const hours = String(date.getHours()).padStart(2, '0');
		const minutes = String(date.getMinutes()).padStart(2, '0');
		const seconds = String(date.getSeconds()).padStart(2, '0');

		return `${day}/${month}/${year} ${hours}:${minutes}:${seconds}`;
	}

	onMount(() => {
		const socket = io(`http://${window.location.hostname}:9000/image_meta`);
		socket.on('recv_image_meta', (data) => {
			if (loading) {
				loading = false;
			}
			// {json of image and metadata} data
			// let current_frame_time = performance.now();
			// push_frame_time(current_frame_time);
			frame = `data:image/jpeg;base64,${data.frame}`;
			metadata = data.metadata;
			if (fps.length > 30) {
				fps.shift();
			}
			fps.push(performance.now());
		});
	});
	// $inspect('pinned_track_ids', pinned_track_ids);
	$inspect('detections', detections);
	// $inspect('metadata', metadata);
	// $inspect('fps_count', fps_count);
</script>

<DemoPage demo={demos['gaze-v2']}>
	{#if loading}
		<div class="flex h-full w-full items-center justify-center">
			<Loader class="text-muted-foreground h-[30%] w-[30%] animate-spin" />
		</div>
	{:else}
		<div class="grid h-full w-full gap-2 px-2 [grid-template-columns:repeat(14,minmax(0,1fr))]">
			<!-- CH: I have no idea why min-h causes the parent to stop from expanding, but it works -->
			<div class="col-span-11 grid h-full min-h-[300px] w-full grid-rows-6 gap-2">
				<div class="row-span-5 h-full w-full">
					<img src={frame} alt="" class="h-full w-full rounded-md" />
				</div>
				<div class="row-span-1 flex h-full w-full gap-2">
					<!-- <GazeLivePercentage
						latest_gaze_people_data={metadata.processed_data.gaze_people_datas[
							metadata.processed_data.gaze_people_datas.length - 1
						]}
					/>
					<div class="h-[200] w-full rounded border p-4">
						<GazeBarChart data={metadata.processed_data.gaze_people_datas} />
					</div> -->
					{#if historical_people_tracks.length > 0}
						<Table.Root>
							<Table.Header>
								<Table.Row>
									<Table.Head>Track Id</Table.Head>
									<Table.Head>Enter Time</Table.Head>
									<Table.Head>Exit Time</Table.Head>
									<Table.Head>Longest Gaze(s)</Table.Head>
									<Table.Head>Gender</Table.Head>
									<Table.Head>Age</Table.Head>
									<Table.Head>Gaze Metric</Table.Head>
									<Table.Head>Score</Table.Head>
									<Table.Head>Face</Table.Head>
								</Table.Row>
							</Table.Header>
							<Table.Body>
								{#each historical_people_tracks as last_detection}
									<Table.Row transitionSlideRight={true}>
										<Table.Head>{last_detection.track_id}</Table.Head>
										<Table.Head>{formatDateTime(last_detection.enter_time)}</Table.Head>
										<Table.Head>{formatDateTime(last_detection.exit_time)}</Table.Head>
										<Table.Head>{last_detection.longest_gaze}</Table.Head>
										<Table.Head>{last_detection.gender}</Table.Head>
										<Table.Head>{last_detection.age}</Table.Head>
										<Table.Head>{last_detection.gaze_metric}</Table.Head>
										<Table.Head>{last_detection.score}</Table.Head>
										<Table.Head>
											<img
												src={`data:image/jpeg;base64,${last_detection.face_img}`}
												alt=""
												class="h-full w-full rounded-md object-fill"
											/>
										</Table.Head>
									</Table.Row>
								{/each}
							</Table.Body>
						</Table.Root>
					{/if}
				</div>
			</div>
			<div class="col-span-3 h-full min-h-[300px] w-full">
				<ScrollArea class="h-full w-full">
					<Card.Root><p>{fps_count}</p></Card.Root>

					{#each detections as detection}
						<Card.Root
							class={cn(
								'w-full',
								['bg-cardBackground', 'text-cardText'],
								pinned_track_ids.includes(detection.track_id)
									? ['border-4', 'border-headerBackground-900']
									: []
							)}
							onclick={() => {
								const track_id = detection.track_id as number;
								if (pinned_track_ids.includes(track_id)) {
									// Unpin the track_id
									pinned_track_ids = pinned_track_ids.filter((tid) => tid !== track_id);
								} else {
									// Pin the track_id
									if (pinned_track_ids.length >= pinned_buffer_size) {
										pinned_track_ids = pinned_track_ids.slice(1, pinned_track_ids.length);
									}
									pinned_track_ids.push(track_id);
								}
							}}
						>
							<div class="grid grid-cols-5 gap-1 p-2">
								<div class="col-span-2 max-h-[120px]">
									<img
										src={`data:image/jpeg;base64,${detection.face_img}`}
										alt=""
										class="h-full w-full rounded-md object-fill"
									/>
								</div>
								<div class="col-span-3">
									<Table.Root class="table-fixed">
										<Table.Body>
											<Table.Row>
												<Table.Cell class="font-medium" pwidth={1}>Track ID</Table.Cell>
												<Table.Cell pwidth={1}>{detection.track_id}</Table.Cell>
											</Table.Row>
											<Table.Row>
												<Table.Cell class="font-medium" pwidth={1}>Gender</Table.Cell>
												<!-- <Table.Cell pwidth={1}
													>{detection.final_gender ? detection.final_gender : 'N.A.'}</Table.Cell
												> -->
												<Table.Cell pwidth={1}>{detection.gender}</Table.Cell>
											</Table.Row>
											<Table.Row>
												<Table.Cell class="font-medium" pwidth={1}>Age</Table.Cell>
												<!-- <Table.Cell pwidth={1}
													>{detection.final_age ? detection.final_age : 'N.A.'}</Table.Cell
												> -->
												<Table.Cell pwidth={1}>{detection.age}</Table.Cell>
											</Table.Row>
											<Table.Row>
												<Table.Cell class="font-medium" pwidth={1}>Gaze</Table.Cell>
												<Table.Cell pwidth={1}
													>{detection.pitch}, {detection.yaw}, {detection.score}</Table.Cell
												>
												<!-- <Table.Cell pwidth={1} class={detection.is_frontal ? 'font-bold' : ''}
													>{detection.is_frontal}</Table.Cell
												> -->
											</Table.Row>
											<Table.Row>
												<Table.Cell class="font-medium" pwidth={1}>ReID</Table.Cell>
												<Table.Cell pwidth={1}>{detection.reid}</Table.Cell>
											</Table.Row>
										</Table.Body>
									</Table.Root>
								</div>
							</div>
						</Card.Root>
					{/each}
				</ScrollArea>
			</div>
		</div>
	{/if}
</DemoPage>
