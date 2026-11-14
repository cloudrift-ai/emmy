//! The lane-to-element table of the tensor-core fragments.
//!
//! A fragment is the registers each lane of a warp holds for one operand or accumulator of a tensor-core product.
//! Which elements of the tile stand in which register of which lane is fixed by the instruction. This module holds
//! that once, per fragment layout and role. A statement that reads or writes a fragment element by element
//! (`FragmentLoad`, `FragmentApply`, `FragmentRowReduce`, `RegStore`, `BlockScaleLoad`) follows from its row of the
//! table, and so does the register count of a `RegFragment`.
//!
//! Only the rows of an accumulator of floats are read by the CUDA renderer; the operand rows, the packed
//! accumulators and the block scales are what its C++ loaders compute, written here for the PTX compiler.

use crate::closed_set;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::parse::Parser;
use crate::statement::RegFragment;
use crate::{Name, Result, value_error};

/// The name the table's coordinates read the thread's lane in its warp by.
pub const LANE: &str = "lane";

/// The selector the block-scaled product takes beside each side's register of scales: which lanes of a group of
/// four supply it. With selector `t`, A's scales come from the lanes whose index in the group is `2t` or `2t + 1`,
/// B's from the lane where it is `t`. Found by probing the hardware; ptxas takes 0 or 1 for either.
pub const SCALE_SELECTOR_A: i64 = 0;
pub const SCALE_SELECTOR_B: i64 = 1;

closed_set! {
    /// A fragment layout: the instruction family whose lane map a fragment follows. A `wgmma` accumulator is, per
    /// warp, the `m16n8k16` one repeated along N, and its register A operand is that layout's.
    pub enum Layout ("fragment layout") {
        M16n8k16 = "m16n8k16", Wgmma = "wgmma", M8n8k4 = "m8n8k4",
    }
}

closed_set! {
    /// What a fragment is to its product: a multiplicand, the accumulator, or one side's register of block scales.
    pub enum Role ("fragment role") {
        A = "a", B = "b", C = "c", ScaleA = "sfa", ScaleB = "sfb",
    }
}

/// One row of the table: what each lane of a warp holds of one fragment.
///
/// Rows and columns are the tile's own: M by K for `a`, K by N for `b`, M by N for `c`, and for block scales the
/// side's own coordinate by the K block. An element is one stored element of the fragment's dtype (a packed pair of
/// 4-bit values is one).
#[derive(Clone, Debug)]
pub struct FragmentTable {
    /// The lane's coordinates the offsets read: a name and its value over [`LANE`] and the coordinates before it.
    pub coordinates: Vec<(Name, Expression)>,
    /// Per register, per element in packing order (the first in the lowest bits): the element's row and column
    /// offset from the tile's base.
    pub registers: Vec<Vec<(Expression, Expression)>>,
    /// The lane XOR masks of the shuffles that fold a row over the lanes that share it. An accumulator's only.
    pub reduce_xors: Vec<i64>,
    /// Which lanes hold the fragment, where not every lane does (a register of block scales).
    pub supplies: Option<Expression>,
}

impl FragmentTable {
    /// The elements one register holds.
    pub fn per_register(&self) -> usize {
        self.registers.first().map_or(0, Vec::len)
    }
}

fn expression(text: &str) -> Expression {
    Parser::new(text, &[])
        .and_then(|mut parser| parser.expression())
        .unwrap_or_else(|err| panic!("the fragment table spells `{text}`: {err}"))
}

/// `base` moved by a constant: the base itself when the constant is zero.
fn offset(base: &str, by: i64) -> Expression {
    match by {
        0 => expression(base),
        by if by > 0 => expression(&format!("{base} + {by}")),
        by => expression(&format!("{base} - {}", -by)),
    }
}

fn coordinates(spelled: &[(&str, &str)]) -> Vec<(Name, Expression)> {
    spelled
        .iter()
        .map(|(name, value)| (Name::new(name), expression(value)))
        .collect()
}

fn registers(
    elements: Vec<(Expression, Expression)>,
    per_register: usize,
) -> Vec<Vec<(Expression, Expression)>> {
    elements
        .chunks(per_register)
        .map(<[(Expression, Expression)]>::to_vec)
        .collect()
}

/// The elements one register of `role` holds at `dtype`: a float or two packed halves for an accumulator, two
/// 16-bit or four 8-bit elements for an operand, the four scales of a lane's K blocks.
fn per_register(role: Role, dtype: DType) -> usize {
    let bytes = matches!(
        dtype,
        DType::F8E4M3 | DType::F8E5M2 | DType::F4E2M1x2 | DType::U8
    );
    match role {
        Role::C if dtype == DType::F16 => 2,
        Role::C => 1,
        Role::A | Role::B if bytes => 4,
        Role::A | Role::B => 2,
        Role::ScaleA | Role::ScaleB => 4,
    }
}

/// The `m16n8` family: a group of four lanes holds one row pair, a lane's place in its group the columns.
const M16N8: [(&str, &str); 2] = [("_g", "lane >> 2"), ("_t", "lane & 3")];

fn m16n8(role: Role, dtype: DType) -> FragmentTable {
    let each = per_register(role, dtype);
    let wide = each as i64;
    let mut table = FragmentTable {
        coordinates: coordinates(&M16N8),
        registers: Vec::new(),
        reduce_xors: Vec::new(),
        supplies: None,
    };
    let mut elements = Vec::new();
    match role {
        Role::C => {
            // Two rows, eight apart, of two adjacent columns each.
            for row in [0, 8] {
                for column in 0..2 {
                    let column = expression(&format!("_t * 2 + {column}"));
                    elements.push((offset("_g", row), column));
                }
            }
            table.reduce_xors = vec![2, 1];
        }
        Role::A => {
            // Register `i`: the row block of its low bit, the K half of its high bit.
            for register in 0..4 {
                for element in 0..wide {
                    let k = (register >> 1) * 4 * wide + element;
                    let row = offset("_g", 8 * (register & 1));
                    elements.push((row, offset(&format!("_t * {wide}"), k)));
                }
            }
        }
        Role::B => {
            for register in 0..2 {
                for element in 0..wide {
                    let k = register * 4 * wide + element;
                    elements.push((offset(&format!("_t * {wide}"), k), expression("_g")));
                }
            }
        }
        Role::ScaleA => {
            let nearer = format!("(lane & 3) - {}", 2 * SCALE_SELECTOR_A);
            table.coordinates = coordinates(&[("_q", "lane >> 2"), ("_r", &nearer)]);
            table.supplies = Some(expression("_r >= 0 && _r <= 1"));
            elements
                .extend((0..4).map(|block| (expression("_q + _r * 8"), Expression::int(block))));
        }
        Role::ScaleB => {
            table.coordinates = coordinates(&[("_q", "lane >> 2")]);
            table.supplies = Some(expression(&format!("(lane & 3) == {SCALE_SELECTOR_B}")));
            elements.extend((0..4).map(|block| (expression("_q"), Expression::int(block))));
        }
    }
    table.registers = registers(elements, each);
    table
}

/// Volta's `m8n8k4`: one warp instruction is four 8 by 8 products, one a group of lanes, laid out as the quadrants
/// of a 16 by 16 tile. `_vq` is the lane's quadrant.
const M8N8K4_QUADRANT: [(&str, &str); 2] = [("_vl", "lane"), ("_vq", "(_vl & 15) >> 2")];
/// The lane's row among the 16 of the tile, as an operand A and a packed accumulator hold it.
const M8N8K4_ROW: &str = "(_vq >> 1) * 8 + (_vl >> 4) * 4 + (_vl & 3)";

fn m8n8k4(role: Role, dtype: DType) -> Result<FragmentTable> {
    let each = per_register(role, dtype);
    let mut spelled = M8N8K4_QUADRANT.to_vec();
    let mut elements = Vec::new();
    let mut reduce_xors = Vec::new();
    match role {
        Role::C if each == 1 => {
            // A lane holds two rows, two apart, and of each two column pairs, four apart.
            spelled.push(("_vr", "(_vq >> 1) * 8 + (_vl >> 4) * 4 + (_vl & 1)"));
            spelled.push(("_vc", "(_vq & 1) * 8 + (_vl & 2)"));
            for element in 0..8 {
                let column = (element & 4) + (element & 1);
                elements.push((offset("_vr", element & 2), offset("_vc", column)));
            }
            reduce_xors = vec![4, 2];
        }
        Role::C => {
            // The packed accumulator holds one row: eight adjacent columns of its quadrant.
            spelled.push(("_vh", M8N8K4_ROW));
            spelled.push(("_vn", "(_vq & 1) * 8"));
            elements.extend((0..8).map(|column| (expression("_vh"), offset("_vn", column))));
            reduce_xors = vec![4];
        }
        Role::A => {
            spelled.push(("_va", M8N8K4_ROW));
            elements.extend((0..4).map(|k| (expression("_va"), Expression::int(k))));
        }
        Role::B => {
            spelled.push(("_vb", "(_vq & 1) * 8 + (_vl >> 4) * 4 + (_vl & 3)"));
            elements.extend((0..4).map(|k| (Expression::int(k), expression("_vb"))));
        }
        Role::ScaleA | Role::ScaleB => {
            return Err(value_error(format!(
                "no fragment role {role} in the layout {}",
                Layout::M8n8k4
            )));
        }
    }
    Ok(FragmentTable {
        coordinates: coordinates(&spelled),
        registers: registers(elements, each),
        reduce_xors,
        supplies: None,
    })
}

/// The row of the table for a fragment of `role` in `layout` whose elements are `dtype`.
pub fn fragment_table(layout: &str, role: &str, dtype: DType) -> Result<FragmentTable> {
    let (layout, role): (Layout, Role) = (layout.parse()?, role.parse()?);
    match (layout, role) {
        (Layout::M8n8k4, role) => m8n8k4(role, dtype),
        // A warp-group product reads B, and block scales, through no register fragment.
        (Layout::Wgmma, Role::B | Role::ScaleA | Role::ScaleB) => Err(value_error(format!(
            "no fragment role {role} in the layout {layout}"
        ))),
        (Layout::M16n8k16 | Layout::Wgmma, role) => Ok(m16n8(role, dtype)),
    }
}

/// The accumulator of floats of one cell of Volta's paired tile: a staged tile whose operands are stored under the
/// paired slab layouts couples two register cells a side into one 32 by 32 map. `cell` is the register cell's
/// index along M and N; its parity picks the half.
pub fn paired_accumulator_table(cell: (i64, i64)) -> FragmentTable {
    let (row_half, column_half) = (-12 * (cell.0 & 1), -12 * (cell.1 & 1));
    let elements = (0..8)
        .map(|element| {
            let row = row_half + (element & 2);
            let column = column_half + 4 * (element & 4) + (element & 1);
            (offset("_vr", row), offset("_vc", column))
        })
        .collect();
    FragmentTable {
        coordinates: coordinates(&[
            ("_vl", "lane"),
            ("_vq", "_vl >> 2"),
            ("_vr", "(((_vq & 4) >> 1) + (_vq & 1)) * 8 + (_vl & 1)"),
            ("_vc", "((_vq >> 1) & 1) * 8 + (_vl & 2)"),
        ]),
        registers: registers(elements, 1),
        reduce_xors: Vec::new(),
        supplies: None,
    }
}

impl RegFragment {
    /// How many registers a lane holds of this fragment: its row of the table says.
    pub fn registers(&self) -> Result<usize> {
        Ok(
            fragment_table(&self.fragment_layout, &self.role, self.dtype)?
                .registers
                .len(),
        )
    }

    /// A lane's register of block scales: one register by the fragment's own name, no array.
    pub fn is_scalar(&self) -> bool {
        matches!(self.role.parse(), Ok(Role::ScaleA | Role::ScaleB))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Num;
    use std::collections::HashMap;

    /// How often each `(row, column)` of the tile is held over the 32 lanes of a warp.
    fn held(table: &FragmentTable) -> HashMap<(i64, i64), usize> {
        let int = |value: &Expression, env: &HashMap<Name, Num>| match value.eval(env).unwrap() {
            Num::Int(value) => value,
            Num::Bool(value) => i64::from(value),
            Num::Float(value) => panic!("{value} in a lane coordinate"),
        };
        let mut held = HashMap::new();
        for lane in 0..32 {
            let mut env = HashMap::from([(Name::new(LANE), Num::Int(lane))]);
            for (name, value) in &table.coordinates {
                let value = int(value, &env);
                env.insert(*name, Num::Int(value));
            }
            if matches!(&table.supplies, Some(supplies) if int(supplies, &env) == 0) {
                continue;
            }
            for (row, column) in table.registers.iter().flatten() {
                *held.entry((int(row, &env), int(column, &env))).or_default() += 1;
            }
        }
        held
    }

    /// Every element of a `rows` by `columns` tile, held `times` times and nothing else.
    fn covers(layout: &str, role: &str, dtype: DType, tile: (i64, i64), times: usize) {
        let table = fragment_table(layout, role, dtype).unwrap();
        let held = held(&table);
        let what = format!("{layout} {role} {dtype}");
        assert_eq!(held.len() as i64, tile.0 * tile.1, "{what}");
        for row in 0..tile.0 {
            for column in 0..tile.1 {
                assert_eq!(
                    held.get(&(row, column)),
                    Some(&times),
                    "{what} ({row}, {column})"
                );
            }
        }
    }

    #[test]
    fn a_warp_holds_every_element_of_the_tile() {
        use DType::*;
        // The m16n8 family: every element once. Bytes pack four a register, so K doubles.
        covers("m16n8k16", "c", F32, (16, 8), 1);
        covers("m16n8k16", "c", F16, (16, 8), 1);
        covers("m16n8k16", "a", F16, (16, 16), 1);
        covers("m16n8k16", "a", BF16, (16, 16), 1);
        covers("m16n8k16", "b", F16, (16, 8), 1);
        covers("m16n8k16", "a", F8E4M3, (16, 32), 1);
        covers("m16n8k16", "b", F8E5M2, (32, 8), 1);
        covers("m16n8k16", "a", F4E2M1x2, (16, 32), 1);
        covers("wgmma", "c", F32, (16, 8), 1);
        covers("wgmma", "a", F16, (16, 16), 1);
        // Block scales: one lane a row of A or a column of B, four K blocks each.
        covers("m16n8k16", "sfa", F4E2M1x2, (16, 4), 1);
        covers("m16n8k16", "sfb", F4E2M1x2, (8, 4), 1);
        // Volta: the accumulator once; an operand twice, by the two quadrants that share it.
        covers("m8n8k4", "c", F32, (16, 16), 1);
        covers("m8n8k4", "c", F16, (16, 16), 1);
        covers("m8n8k4", "a", F16, (16, 4), 2);
        covers("m8n8k4", "b", F16, (4, 16), 2);
    }

    /// The layouts and roles are spelled in stored statement text.
    #[test]
    fn the_layouts_and_roles_are_spelled() {
        let layouts: Vec<&str> = Layout::ALL.iter().map(|layout| layout.text()).collect();
        assert_eq!(layouts, ["m16n8k16", "wgmma", "m8n8k4"]);
        let roles: Vec<&str> = Role::ALL.iter().map(|role| role.text()).collect();
        assert_eq!(roles, ["a", "b", "c", "sfa", "sfb"]);
    }

    #[test]
    fn the_register_count_is_the_tables() {
        let count = |layout: &str, role: &str, dtype: DType| {
            fragment_table(layout, role, dtype)
                .map(|table| (table.registers.len(), table.per_register()))
        };
        use DType::*;
        assert_eq!(count("m16n8k16", "a", F16).unwrap(), (4, 2));
        assert_eq!(count("m16n8k16", "b", BF16).unwrap(), (2, 2));
        assert_eq!(count("m16n8k16", "c", F32).unwrap(), (4, 1));
        assert_eq!(count("m16n8k16", "c", F16).unwrap(), (2, 2));
        assert_eq!(count("m16n8k16", "a", F8E4M3).unwrap(), (4, 4));
        assert_eq!(count("m16n8k16", "a", F4E2M1x2).unwrap(), (4, 4));
        assert_eq!(count("m16n8k16", "b", F4E2M1x2).unwrap(), (2, 4));
        assert_eq!(count("m16n8k16", "sfa", F4E2M1x2).unwrap(), (1, 4));
        assert_eq!(count("wgmma", "a", F16).unwrap(), (4, 2));
        assert_eq!(count("wgmma", "c", F32).unwrap(), (4, 1));
        assert_eq!(count("m8n8k4", "a", F16).unwrap(), (2, 2));
        assert_eq!(count("m8n8k4", "b", F16).unwrap(), (2, 2));
        assert_eq!(count("m8n8k4", "c", F32).unwrap(), (8, 1));
        assert_eq!(count("m8n8k4", "c", F16).unwrap(), (4, 2));
        for (layout, role) in [
            ("wgmma", "b"),
            ("m8n8k4", "sfa"),
            ("m16n8k32", "a"),
            ("m16n8k16", "d"),
        ] {
            assert!(count(layout, role, F16).is_err(), "{layout} {role}");
        }
    }

    /// The paired Volta tile: the four cells of a 2 by 2 register tile hold a 32 by 32 tile once between them.
    #[test]
    fn the_cells_of_a_paired_tile_hold_it_once() {
        let mut held_by_all: HashMap<(i64, i64), usize> = HashMap::new();
        for cell in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            for ((row, column), times) in held(&paired_accumulator_table(cell)) {
                *held_by_all
                    .entry((row + 16 * cell.0, column + 16 * cell.1))
                    .or_default() += times;
            }
        }
        assert_eq!(held_by_all.len(), 32 * 32);
        assert!(held_by_all.iter().all(|((row, column), times)| {
            (0..32).contains(row) && (0..32).contains(column) && *times == 1
        }));
    }
}
