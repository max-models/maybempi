// @ts-check
// The shared max-models docs setup (@max-models/starlight-theme): served at
// https://max-models.github.io/maybempi/, with math, the API reference and a link check.
import { defineDocs } from '@max-models/starlight-theme';

export default defineDocs({
	repo: 'maybempi',
	title: 'maybempi',
	description: 'Use MPI only when launched under MPI, and a serial stand-in for mpi4py otherwise.',
	pythonPackage: 'maybempi',
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'Guides', link: '/guides/detection/' },
		{ label: 'Tutorials', link: '/tutorials/' },
		{ label: 'Development', link: '/development/docs/' },
		{ label: 'API', link: '/api/maybempi/' },
	],
	sidebar: [
		{
			label: 'Getting started',
			items: [
				{ label: 'Installation', slug: 'getting-started/installation' },
				{ label: 'Quickstart', slug: 'getting-started/quickstart' },
			],
		},
		{
			label: 'Guides',
			items: [
				{ label: 'How the decision is made', slug: 'guides/detection' },
				{ label: 'The serial stand-in', slug: 'guides/serial-stand-in' },
				{ label: 'Using maybempi in a library', slug: 'guides/libraries' },
			],
		},
		{
			label: 'Tutorials',
			items: [{ autogenerate: { directory: 'tutorials' } }],
		},
		{
			label: 'Development',
			items: [
				{ label: 'Building the docs', slug: 'development/docs' },
				{ label: 'Publishing', slug: 'development/publishing' },
			],
		},
	],
});
