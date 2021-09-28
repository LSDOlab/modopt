# user api
echo "---\ntitle: Model\nsidebar_position: 1\n---\n\n------------------------------------------------------------------------\n\n" > ../docs/lang_ref/model.mdx
pydoc-markdown -m modopt.core.model user.yml >> ../docs/lang_ref/model.mdx
sed -i -e 's/#### /### /g' ../docs/lang_ref/model.mdx

echo "---\ntitle: Output\nsidebar_position: 2\n---\n\n------------------------------------------------------------------------\n\n" > ../docs/lang_ref/output.mdx
pydoc-markdown -m modopt.core.explicit_output user.yml >> ../docs/lang_ref/output.mdx
echo "\n\n" >> ../docs/lang_ref/custom.mdx
pydoc-markdown -m modopt.core.implicit_output user.yml >> ../docs/lang_ref/output.mdx
sed -i -e 's/#### /### /g' ../docs/lang_ref/implicit_output.mdx

echo "---\ntitle: SimulatorBase\nsidebar_position: 2\n---\n\n------------------------------------------------------------------------\n\n" > ../docs/lang_ref/simulator_base.mdx
pydoc-markdown -m modopt.core.simulator_base user.yml >> ../docs/lang_ref/simulator_base.mdx
sed -i -e 's/#### /### /g' ../docs/lang_ref/simulator_base.mdx

echo "---\ntitle: Custom Operations\nsidebar_position: 3\n---\n\n------------------------------------------------------------------------\n\n" > ../docs/lang_ref/custom.mdx
pydoc-markdown -m modopt.core.explicit_operation user.yml >> ../docs/lang_ref/custom.mdx
echo "\n\n" >> ../docs/lang_ref/custom.mdx
pydoc-markdown -m modopt.core.implicit_operation user.yml >>../docs/lang_ref/custom.mdx
sed -i -e 's/#### /### /g' ../docs/lang_ref/custom.mdx

# developer api
echo "---\ntitle: Developer API\n---\n\n------------------------------------------------------------------------\n\n" > ../docs/developer/api.mdx
pydoc-markdown -m modopt.core.model dev.yml >> ../docs/developer/api/model.mdx
echo "\n\n" >> ../docs/developer/api.mdx
pydoc-markdown -m modopt.core.simulator_base dev.yml >> ../docs/developer/api.mdx
echo "\n\n" >> ../docs/developer/api.mdx
pydoc-markdown -m modopt.core.explicit_operation dev.yml >> ../docs/developer/api.mdx
echo "\n\n" >> ../docs/developer/api.mdx
pydoc-markdown -m modopt.core.implicit_operation dev.yml >>../docs/developer/api.mdx
echo "\n\n" >> ../docs/developer/api.mdx
pydoc-markdown -m modopt.core.node dev.yml >> ../docs/developer/api.mdx
echo "\n\n" >> ../docs/developer/api.mdx
pydoc-markdown -m modopt.core.variable dev.yml >> ../docs/developer/api.mdx
echo "\n\n" >> ../docs/developer/api.mdx
pydoc-markdown -m modopt.core.output dev.yml >> ../docs/developer/api.mdx
echo "\n\n" >> ../docs/developer/api.mdx
pydoc-markdown -m modopt.core.explicit_output dev.yml >> ../docs/developer/api.mdx
echo "\n\n" >> ../docs/developer/api.mdx
pydoc-markdown -m modopt.core.implicit_output dev.yml >> ../docs/developer/api.mdx
echo "\n\n" >> ../docs/developer/api.mdx
pydoc-markdown -m modopt.core.operation dev.yml >> ../docs/developer/api.mdx
echo "\n\n" >> ../docs/developer/api.mdx
pydoc-markdown -m modopt.core.standard_operation dev.yml >> ../docs/developer/api.mdx
echo "\n\n" >> ../docs/developer/api.mdx
pydoc-markdown -m modopt.core.custom_operation dev.yml >> ../docs/developer/api.mdx
echo "\n\n" >> ../docs/developer/api.mdx
pydoc-markdown -m modopt.core.subgraph dev.yml >> ../docs/developer/api.mdx
sed -i -e 's/#### /### /g' ../docs/developer/api.mdx