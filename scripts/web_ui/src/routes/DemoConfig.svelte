<script lang="ts">
	import { Button } from '$lib/components/ui/button/index';
	import { demos, type Demo } from '$lib/demos';
	import { getPaths } from '$lib/helpers';

	let demo = $derived.by(() => {
		return Object.values(demos).find((demo) => getPaths().some((path) => path === demo.value));
	}) as Demo;
</script>

<Button variant="link" href={`${demo.path}/config`} class="text-headerText font-bold">Demo Config</Button>
