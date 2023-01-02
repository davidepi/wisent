mod conversion;
mod ll;

pub use self::ll::first_follow;

/// Represents the empty value `ε` when appearing in the first set or follow set.
pub const EPSILON_VAL: u32 = 0xFFFFFFFE;
/// Represents the end of line value `$` when appearing in the follow set.
pub const ENDLINE_VAL: u32 = 0xFFFFFFFD;
