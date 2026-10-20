/// What POST /plaid/refresh reports. Parsing is deliberately forgiving: a
/// missing field becomes null or zero and an unrecognised field is ignored,
/// so a later backend addition cannot break the app.
enum PlaidRefreshStatus {
  refreshed,
  partial,
  failed,
  skippedRecent,
  noConnections,

  /// A status this build does not know. Treated like [failed] for messaging.
  unknown;

  static PlaidRefreshStatus parse(Object? raw) => switch (raw) {
        'refreshed' => refreshed,
        'partial' => partial,
        'failed' => failed,
        'skipped_recent' => skippedRecent,
        'no_connections' => noConnections,
        _ => unknown,
      };
}

/// One Item's outcome inside a refresh.
class PlaidItemRefresh {
  const PlaidItemRefresh({
    required this.itemId,
    this.institutionName,
    required this.refreshed,
    this.errorCode,
    this.added = 0,
    this.modified = 0,
    this.removed = 0,
    this.accountsUpdated = 0,
    this.investmentsSynced = false,
  });

  final String itemId;
  final String? institutionName;

  /// The Item's own status was "refreshed". Anything else counts as failed.
  final bool refreshed;

  /// Plaid's error_code when the Item failed, e.g. ITEM_LOGIN_REQUIRED.
  final String? errorCode;
  final int added;
  final int modified;
  final int removed;
  final int accountsUpdated;
  final bool investmentsSynced;

  factory PlaidItemRefresh.fromJson(Map<String, dynamic> json) {
    final txns = json['transactions'];
    int count(String key) =>
        txns is Map ? (txns[key] as num?)?.toInt() ?? 0 : 0;
    return PlaidItemRefresh(
      itemId: (json['item_id'] as String?) ?? '',
      institutionName: json['institution_name'] as String?,
      refreshed: json['status'] == 'refreshed',
      errorCode: json['error_code'] as String?,
      added: count('added'),
      modified: count('modified'),
      removed: count('removed'),
      accountsUpdated: (json['accounts_updated'] as num?)?.toInt() ?? 0,
      investmentsSynced: json['investments_synced'] == true,
    );
  }
}

class PlaidRefreshResult {
  const PlaidRefreshResult({
    required this.status,
    this.refreshedAt,
    this.items = const [],
  });

  final PlaidRefreshStatus status;

  /// When the server finished. Null on skipped_recent, which carries the
  /// previous sync time instead and is not a refresh.
  final DateTime? refreshedAt;
  final List<PlaidItemRefresh> items;

  List<PlaidItemRefresh> get failedItems =>
      [for (final item in items) if (!item.refreshed) item];

  factory PlaidRefreshResult.fromJson(Map<String, dynamic> json) {
    final rawItems = json['items'];
    final rawTime = json['refreshed_at'];
    return PlaidRefreshResult(
      status: PlaidRefreshStatus.parse(json['status']),
      refreshedAt: rawTime is String ? DateTime.tryParse(rawTime) : null,
      items: List.unmodifiable([
        if (rawItems is List)
          for (final item in rawItems)
            if (item is Map)
              PlaidItemRefresh.fromJson(Map<String, dynamic>.from(item)),
      ]),
    );
  }
}
