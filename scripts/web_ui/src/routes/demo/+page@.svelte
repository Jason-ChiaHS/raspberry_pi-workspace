<script lang="ts">
	import ModeSwitch from './ModeSwitch.svelte';
	import DemoSelector from './DemoSelector.svelte';
	import io from 'socket.io-client';
	import { onMount } from 'svelte';
	import { APP_VERSION } from '$lib/vars';
	import * as Resizable from '$lib/components/ui/resizable/index.js';

	let frame = $state('');
	let metadata = $state('');
	let display_metadata = $derived(
		metadata === '' ? 'Waiting for Demo to start...' : JSON.stringify(metadata)
	);

	let frame_times: number[] = $state([]);
	let fps = $derived(
		frame_times.length > 15
			? `FPS: ${(frame_times.length / (frame_times[frame_times.length - 1] - frame_times[0])) * 1000}`
			: 'Waiting for Demo to start...'
	);

	function push_frame_time(frame_time: number) {
		frame_times.push(frame_time);
		if (frame_times.length > 30) {
			frame_times.shift();
		}
	}

	onMount(() => {
		const socket = io(`http://${window.location.hostname}:9000/image_meta`);
		socket.on('recv_image_meta', (data) => {
			// {json of image and metadata} data
			let current_frame_time = performance.now();
			push_frame_time(current_frame_time);
			frame = `data:image/jpeg;base64,${data.frame}`;
			metadata = data.metadata;
		});
	});
</script>

<div class="h-dvh">
	<div class="grid h-full w-full grid-cols-5 grid-rows-5 gap-1 md:p-2">
		<div class="col-span-1 row-span-5 flex flex-col gap-1 rounded-md border-2 p-1">
			<div class="flex items-center justify-center border-2">
				<p class="d-block m-2 inline">Pentas Demo Hub v{APP_VERSION}</p>
				<ModeSwitch />
			</div>
			<Resizable.PaneGroup direction="vertical" class="flex-grow border-2">
				<Resizable.Pane>
					<DemoSelector />
				</Resizable.Pane>
				<Resizable.Handle withHandle />
				<Resizable.Pane>
					<textarea class="h-full w-full resize-none" bind:value={display_metadata}> </textarea>
				</Resizable.Pane>
			</Resizable.PaneGroup>
			<div class="border-2">
				<p>{fps}</p>
			</div>
		</div>
		<div class="relative col-span-4 row-span-5 rounded-md border-2 p-2">
			<img alt="cat" class="absolute left-0 top-0 h-full w-full" src={frame} />
		</div>
	</div>
</div>
