<script lang="ts">
    import DemoPage from "$lib/pages/DemoPage/index.svelte";
    import { demos } from "$lib/demos";
	import { Loader } from '@lucide/svelte';

	import io from 'socket.io-client';
	import { onMount } from 'svelte';

	let loading = $state(true);
	let frame = $state('');
	let metadata: { [key: string]: any } = $state({});
	let fps: number[] = $state([]);
	let fps_count = $derived.by(() => {
		if (fps.length > 2) {
			return ((fps.length - 1) / (fps[fps.length - 1] - fps[0])) * 1000;
		} else {
			return 'Need more frames';
		}
	});

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
</script>

<DemoPage demo={demos["helmet"]}>
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
			</div>
		</div>
	{/if}
</DemoPage>
